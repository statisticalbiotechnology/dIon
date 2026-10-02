"""Test whether a fixed A+B mixture follows its supplied precursor query."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import yaml
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.distractor_augmentation import BatchedStudentDistractorMixAugmentation
from src.embed_eval.counterfactual import (
    evaluate_counterfactual_source_selection,
    select_counterfactual_pairs,
)
from src.embed_eval.data import PeptideRetrievalDataset, build_embedding_eval_collate
from src.embed_eval.dense_counterfactual import evaluate_dense_precursor_source_selection
from src.embed_eval.extraction import preserve_eval_mode
from src.embed_eval.loading import load_checkpoint_embedder, load_checkpoint_teacher_backbone
from src.parse_args import parse_args_and_config


def _parse_script_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dense_peak_tokens", action="store_true")
    parser.add_argument("--dense_counterfactual_output_report", default="")
    parser.add_argument("--dense_counterfactual_max_pairs_per_partition", type=int, default=None)
    parser.add_argument("--dense_counterfactual_batch_size", type=int, default=None)
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    return args


class CounterfactualPairDataset(Dataset):
    """Expose selected retrieval rows as ordered A/B counterfactual pairs."""

    def __init__(self, dataset, records) -> None:
        self.dataset = dataset
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        anchor = self.dataset[record.anchor_index]
        distractor = self.dataset[record.distractor_index]
        anchor["_counterfactual_pair_index"] = index
        anchor["_counterfactual_side"] = 0
        distractor["_counterfactual_pair_index"] = index
        distractor["_counterfactual_side"] = 1
        return anchor, distractor


def _stable_seed(seed: int, *parts: object) -> int:
    payload = "\x1f".join([str(seed), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big") % (2**31)


def _build_pair_collate(global_args):
    collate = build_embedding_eval_collate(global_args)

    def collate_pairs(items):
        return collate([spectrum for pair in items for spectrum in pair])

    return collate_pairs


def _complete_pair_positions(batch):
    pair_indices = batch["_counterfactual_pair_index"].reshape(-1).tolist()
    sides = batch["_counterfactual_side"].reshape(-1).tolist()
    positions = {}
    for position, (pair_index, side) in enumerate(zip(pair_indices, sides, strict=True)):
        positions.setdefault(int(pair_index), {})[int(side)] = position
    complete = [pair_index for pair_index in sorted(positions) if {0, 1} <= positions[pair_index].keys()]
    if not complete:
        return None, None, None
    return (
        torch.tensor([positions[pair_index][0] for pair_index in complete], dtype=torch.long),
        torch.tensor([positions[pair_index][1] for pair_index in complete], dtype=torch.long),
        complete,
    )


def _batch_inputs(batch, positions, device):
    mz = batch["mz_array"].index_select(0, positions).to(device)
    intensity = batch["intensity_array"].index_select(0, positions).to(device)
    peaks = torch.stack([mz, intensity], dim=-1)
    lengths = batch["peak_lengths"].reshape(-1).index_select(0, positions).to(device)
    padding = torch.arange(peaks.shape[1], device=device).unsqueeze(0).ge(lengths.unsqueeze(1))
    return {
        "peaks": peaks,
        "padding": padding,
        "mass": batch["precursor_mass"].reshape(-1).index_select(0, positions).to(device),
        "charge": batch["precursor_charge"].reshape(-1).index_select(0, positions).to(device),
    }


def _embed(embedder, inputs):
    return embedder(
        inputs["peaks"],
        inputs["padding"],
        inputs["mass"] if embedder.use_mass else None,
        inputs["charge"] if embedder.use_charge else None,
    )


def _merge_clean(mixer, inputs, source_label):
    provenance = torch.full_like(inputs["padding"], source_label, dtype=torch.int8)
    provenance.masked_fill_(inputs["padding"], mixer.PADDING)
    peaks, padding, _ = mixer.merge_batched(inputs["peaks"], inputs["padding"], provenance)
    return {**inputs, "peaks": peaks, "padding": padding}


def _encode_dense(encoder, inputs):
    encoded = encoder(
        inputs["peaks"],
        key_padding_mask=inputs["padding"],
        mass=inputs["mass"] if getattr(encoder, "use_mass", False) else None,
        charge=inputs["charge"] if getattr(encoder, "use_charge", False) else None,
    )
    prefix = int(encoded["num_cem_tokens"])
    mask = encoded.get("mask")
    if mask is None:
        mask = torch.zeros(
            encoded["emb"].shape[:2], dtype=torch.bool, device=encoded["emb"].device
        )
    return encoded["emb"][:, prefix:], mask[:, prefix:]


def _match_source_tokens(mixed_mz, mixed_padding, provenance, clean_mz, clean_padding, label, ppm):
    mixed_positions = torch.nonzero((~mixed_padding) & provenance.eq(label), as_tuple=False).flatten()
    clean_positions = torch.nonzero(~clean_padding, as_tuple=False).flatten()
    if mixed_positions.numel() == 0 or clean_positions.numel() == 0:
        return None, None
    errors = (mixed_mz[mixed_positions, None] - clean_mz[clean_positions][None, :]).abs()
    errors = errors / clean_mz[clean_positions][None, :].abs().clamp_min(1e-8) * 1e6
    nearest_error, nearest = errors.min(dim=1)
    keep = nearest_error.le(ppm)
    mixed_positions = mixed_positions[keep]
    clean_positions = clean_positions[nearest[keep]]
    if mixed_positions.numel() == 0:
        return None, None
    unique = torch.tensor(
        [clean_positions.eq(value).sum().eq(1) for value in clean_positions],
        device=clean_positions.device,
        dtype=torch.bool,
    )
    mixed_positions, clean_positions = mixed_positions[unique], clean_positions[unique]
    return (mixed_positions, clean_positions) if mixed_positions.numel() else (None, None)


def _cosine(left, right):
    return F.cosine_similarity(left.float(), right.float(), dim=-1)


def main() -> None:
    script_args = _parse_script_args()
    global_args, pretrain_config, _, probing_config = parse_args_and_config()
    suite_key = (
        "dense_precursor_counterfactual_evaluation"
        if script_args.dense_peak_tokens
        else "precursor_counterfactual_evaluation"
    )
    if not probing_config or suite_key not in probing_config:
        raise ValueError(f"--probing_config must contain {suite_key!r}.")
    if global_args.embedding_baseline != "model":
        raise ValueError("Precursor counterfactual evaluation requires model embeddings.")
    if script_args.dense_peak_tokens:
        _run_dense(global_args, pretrain_config, probing_config[suite_key], script_args)
        return
    if global_args.counterfactual_distance == "jensen_shannon":
        if global_args.embedding_readout != "dino_logits":
            raise ValueError(
                "Jensen-Shannon counterfactual evaluation requires "
                "--embedding_readout dino_logits."
            )
    elif global_args.embedding_readout == "dino_logits":
        raise ValueError(
            "--embedding_readout dino_logits is only supported with "
            "--counterfactual_distance jensen_shannon."
        )
    suite = probing_config[suite_key]
    retrieval_path = Path(suite["retrieval_config_path"])
    retrieval_root = yaml.safe_load(retrieval_path.read_text()) or {}
    retrieval_config = retrieval_root["embedding_evaluation"]
    dataset_cfg = retrieval_config["dataset"]
    selection_cfg = retrieval_config.get("selection", {})
    dataset = PeptideRetrievalDataset(
        dataset_cfg["parquet_path"],
        peptide_id_column=dataset_cfg.get("peptide_id_column", "peptide_id"),
        partition_column=dataset_cfg.get("partition_column", "species"),
        seed=int(selection_cfg.get("seed", 0)),
        max_peptides_per_partition=selection_cfg.get("max_peptides_per_partition"),
        max_spectra_per_peptide=selection_cfg.get("max_spectra_per_peptide"),
    )
    mixing = suite["mixing"]
    mixer = BatchedStudentDistractorMixAugmentation(
        mix_apply_to="all",
        condition_separation_ppm=float(mixing.get("condition_separation_ppm", 10.0)),
        neutral_mass_separation_ppm=float(mixing.get("neutral_mass_separation_ppm", 10.0)),
        merge_ppm=float(mixing.get("merge_ppm", 5.0)),
        intensity_normalization="none",
    )
    records, selection_summary = select_counterfactual_pairs(
        dataset,
        mixer=mixer,
        seed=int(suite.get("seed", 0)),
        max_pairs_per_partition=suite.get("max_pairs_per_partition"),
    )
    pair_dataset = CounterfactualPairDataset(dataset, records)
    extraction = suite.get("extraction", {})
    device = torch.device("cuda" if global_args.accelerator == "gpu" and torch.cuda.is_available() else "cpu")
    _, embedder = load_checkpoint_embedder(global_args, pretrain_config)
    loader = DataLoader(
        pair_dataset,
        batch_size=int(extraction.get("batch_size", 128)),
        shuffle=False,
        num_workers=int(extraction.get("num_workers", 0)),
        pin_memory=bool(extraction.get("pin_memory", False)) and device.type == "cuda",
        collate_fn=_build_pair_collate(global_args),
    )
    clean_a_values, clean_b_values, mixed_a_values, mixed_b_values = [], [], [], []
    partition_ids = []
    processed_pairs = 0
    embedder = embedder.to(device)
    base_seed = int(suite.get("seed", 0))
    with preserve_eval_mode(embedder), torch.inference_mode():
        for batch in tqdm(loader, desc="Counterfactual A+B", unit="batch", dynamic_ncols=True):
            anchor_positions, distractor_positions, complete_pair_ids = _complete_pair_positions(batch)
            if anchor_positions is None:
                continue
            anchor = _batch_inputs(batch, anchor_positions, device)
            distractor = _batch_inputs(batch, distractor_positions, device)
            cuda_devices = [device] if device.type == "cuda" else []
            with torch.random.fork_rng(devices=cuda_devices):
                torch.manual_seed(_stable_seed(base_seed, "mix", *complete_pair_ids))
                mixed_peaks, mixed_padding = mixer.mix_fixed_distractors(
                    anchor["peaks"], anchor["padding"],
                    distractor["peaks"], distractor["padding"],
                    strength=float(mixing.get("distractor_peak_fraction", 0.5)),
                )
            mixed_anchor = {**anchor, "peaks": mixed_peaks, "padding": mixed_padding}
            mixed_distractor = {**distractor, "peaks": mixed_peaks, "padding": mixed_padding}
            clean_a = _embed(embedder, anchor)
            clean_b = _embed(embedder, distractor)
            mixed_a = _embed(embedder, mixed_anchor)
            mixed_b = _embed(embedder, mixed_distractor)
            if global_args.counterfactual_distance == "jensen_shannon":
                clean_a, clean_b, mixed_a, mixed_b = [
                    torch.softmax(values.float(), dim=1)
                    for values in (clean_a, clean_b, mixed_a, mixed_b)
                ]
            clean_a_values.append(clean_a.cpu())
            clean_b_values.append(clean_b.cpu())
            mixed_a_values.append(mixed_a.cpu())
            mixed_b_values.append(mixed_b.cpu())
            partition_ids.extend(
                str(batch["partition_id"][index]) for index in anchor_positions.tolist()
            )
            processed_pairs += anchor_positions.numel()
    if processed_pairs != len(records):
        raise RuntimeError(
            f"Only {processed_pairs}/{len(records)} pairs survived preprocessing; "
            "counterfactual evaluation requires complete A/B pairs."
        )
    metrics = evaluate_counterfactual_source_selection(
        torch.cat(clean_a_values), torch.cat(clean_b_values),
        torch.cat(mixed_a_values), torch.cat(mixed_b_values), partition_ids,
        distance_metric=global_args.counterfactual_distance,
    )
    report = {
        "checkpoint_path": global_args.encoder_weights,
        "embedding_readout": global_args.embedding_readout,
        "distance_metric": global_args.counterfactual_distance,
        "representation": (
            "EMA-teacher pooled backbone features"
            if global_args.embedding_readout == "backbone"
            else "EMA-teacher full DINO prototype-head logits transformed with softmax"
        ),
        "protocol": "fixed_mixture_precursor_swap",
        "description": "The same A+B peak mixture is embedded once with A precursor metadata and once with B precursor metadata; each is compared with both clean source embeddings.",
        "dataset": {"name": retrieval_config["name"], "selection": dataset.selection.__dict__},
        "mixing": mixing,
        "selection": selection_summary,
        "metrics": metrics,
    }
    configured = global_args.counterfactual_output_report or suite.get("output", {}).get("report_path")
    output_path = Path(configured) if configured else Path(global_args.encoder_weights).with_suffix(".precursor_counterfactual.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"selection": selection_summary, "macro": metrics["macro"], "pooled": metrics["pooled"]}, indent=2, sort_keys=True))
    print(f"Wrote precursor counterfactual report: {output_path}")



def _run_dense(global_args, pretrain_config, suite, script_args) -> None:
    """Run the fixed-mixture protocol using final EMA peak tokens."""
    retrieval_root = yaml.safe_load(Path(suite["retrieval_config_path"]).read_text()) or {}
    retrieval_config = retrieval_root["embedding_evaluation"]
    dataset_cfg = dict(retrieval_config["dataset"])
    dataset_cfg["parquet_path"] = str(
        suite.get("retrieval_parquet_path", dataset_cfg["parquet_path"])
    )
    selection_cfg = retrieval_config.get("selection", {})
    dataset = PeptideRetrievalDataset(
        dataset_cfg["parquet_path"],
        peptide_id_column=dataset_cfg.get("peptide_id_column", "peptide_id"),
        partition_column=dataset_cfg.get("partition_column", "species"),
        seed=int(selection_cfg.get("seed", 0)),
        max_peptides_per_partition=selection_cfg.get("max_peptides_per_partition"),
        max_spectra_per_peptide=selection_cfg.get("max_spectra_per_peptide"),
    )
    mixing = suite["mixing"]
    mixer = BatchedStudentDistractorMixAugmentation(
        mix_apply_to="all",
        condition_separation_ppm=float(mixing.get("condition_separation_ppm", 10.0)),
        neutral_mass_separation_ppm=float(mixing.get("neutral_mass_separation_ppm", 10.0)),
        merge_ppm=float(mixing.get("merge_ppm", 5.0)),
        intensity_normalization="none",
    )
    max_pairs = script_args.dense_counterfactual_max_pairs_per_partition
    if max_pairs is None:
        max_pairs = suite.get("max_pairs_per_partition")
    records, selection_summary = select_counterfactual_pairs(
        dataset,
        mixer=mixer,
        seed=int(suite.get("seed", 0)),
        max_pairs_per_partition=max_pairs,
    )
    device = torch.device(
        "cuda" if global_args.accelerator == "gpu" and torch.cuda.is_available() else "cpu"
    )
    encoder = load_checkpoint_teacher_backbone(global_args, pretrain_config)
    extraction = suite.get("extraction", {})
    batch_size = script_args.dense_counterfactual_batch_size or int(
        extraction.get("batch_size", 128)
    )
    loader = DataLoader(
        CounterfactualPairDataset(dataset, records),
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(extraction.get("num_workers", 0)),
        pin_memory=bool(extraction.get("pin_memory", False)) and device.type == "cuda",
        collate_fn=_build_pair_collate(global_args),
    )
    matching_ppm = float(suite.get("matching", {}).get("source_peak_match_ppm", 0.01))
    values = {name: [] for name in ("a_adv", "b_adv", "a_dist", "b_dist", "a_count", "b_count")}
    partition_ids = []
    skipped = {"incomplete_pair": 0, "missing_anchor_match": 0, "missing_distractor_match": 0}
    provenance_counts = {"anchor_only": 0, "distractor_only": 0, "merged": 0, "padding": 0}
    encoder = encoder.to(device)
    base_seed = int(suite.get("seed", 0))
    with preserve_eval_mode(encoder), torch.inference_mode():
        for batch in tqdm(loader, desc="Dense counterfactual A+B", unit="batch", dynamic_ncols=True):
            a_positions, b_positions, pair_ids = _complete_pair_positions(batch)
            if a_positions is None:
                skipped["incomplete_pair"] += 1
                continue
            anchor = _batch_inputs(batch, a_positions, device)
            distractor = _batch_inputs(batch, b_positions, device)
            with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
                torch.manual_seed(_stable_seed(base_seed, "mix", *pair_ids))
                mixed_peaks, mixed_padding, provenance = mixer.mix_fixed_distractors(
                    anchor["peaks"], anchor["padding"],
                    distractor["peaks"], distractor["padding"],
                    strength=float(mixing.get("distractor_peak_fraction", 0.5)),
                    return_provenance=True,
                )
            for label, name in (
                (mixer.ANCHOR, "anchor_only"),
                (mixer.DISTRACTOR, "distractor_only"),
                (mixer.MIXED, "merged"),
                (mixer.PADDING, "padding"),
            ):
                provenance_counts[name] += int(provenance.eq(label).sum().item())
            clean_a = _merge_clean(mixer, anchor, mixer.ANCHOR)
            clean_b = _merge_clean(mixer, distractor, mixer.DISTRACTOR)
            mixed_a = {**anchor, "peaks": mixed_peaks, "padding": mixed_padding}
            mixed_b = {**distractor, "peaks": mixed_peaks, "padding": mixed_padding}
            clean_a_tokens, _ = _encode_dense(encoder, clean_a)
            clean_b_tokens, _ = _encode_dense(encoder, clean_b)
            mixed_a_tokens, mixed_a_mask = _encode_dense(encoder, mixed_a)
            mixed_b_tokens, mixed_b_mask = _encode_dense(encoder, mixed_b)
            if not torch.equal(mixed_a_mask, mixed_b_mask):
                raise RuntimeError("Identical mixed peaks produced different dense padding masks.")
            for row in range(mixed_peaks.shape[0]):
                a_mixed, a_clean = _match_source_tokens(
                    mixed_peaks[row, :, 0], mixed_padding[row], provenance[row],
                    clean_a["peaks"][row, :, 0], clean_a["padding"][row],
                    mixer.ANCHOR, matching_ppm,
                )
                b_mixed, b_clean = _match_source_tokens(
                    mixed_peaks[row, :, 0], mixed_padding[row], provenance[row],
                    clean_b["peaks"][row, :, 0], clean_b["padding"][row],
                    mixer.DISTRACTOR, matching_ppm,
                )
                if a_mixed is None:
                    skipped["missing_anchor_match"] += 1
                    continue
                if b_mixed is None:
                    skipped["missing_distractor_match"] += 1
                    continue
                a_under_a = _cosine(mixed_a_tokens[row, a_mixed], clean_a_tokens[row, a_clean])
                a_under_b = _cosine(mixed_b_tokens[row, a_mixed], clean_a_tokens[row, a_clean])
                b_under_b = _cosine(mixed_b_tokens[row, b_mixed], clean_b_tokens[row, b_clean])
                b_under_a = _cosine(mixed_a_tokens[row, b_mixed], clean_b_tokens[row, b_clean])
                values["a_adv"].append((a_under_a - a_under_b).mean().cpu())
                values["b_adv"].append((b_under_b - b_under_a).mean().cpu())
                values["a_dist"].append(
                    (1.0 - _cosine(mixed_a_tokens[row, a_mixed], mixed_b_tokens[row, a_mixed])).mean().cpu()
                )
                values["b_dist"].append(
                    (1.0 - _cosine(mixed_a_tokens[row, b_mixed], mixed_b_tokens[row, b_mixed])).mean().cpu()
                )
                values["a_count"].append(torch.tensor(a_mixed.numel()))
                values["b_count"].append(torch.tensor(b_mixed.numel()))
                partition_ids.append(str(batch["partition_id"][a_positions[row].item()]))
    if not partition_ids:
        raise RuntimeError("No mixture retained uniquely matched A-only and B-only peak tokens.")
    metrics = evaluate_dense_precursor_source_selection(
        torch.stack(values["a_adv"]), torch.stack(values["b_adv"]),
        torch.stack(values["a_dist"]), torch.stack(values["b_dist"]),
        torch.stack(values["a_count"]), torch.stack(values["b_count"]), partition_ids,
    )
    report = {
        "checkpoint_path": str(Path(global_args.encoder_weights).resolve()),
        "representation": "EMA-teacher final dense peak tokens",
        "protocol": "fixed_mixture_precursor_swap",
        "description": "Same A+B mixture under A and B precursor; source-only peak tokens are compared with uniquely m/z-matched clean-source tokens. Merged peaks are excluded.",
        "dataset": {"name": retrieval_config["name"], "selection": dataset.selection.__dict__},
        "mixing": mixing,
        "matching": {"source_peak_match_ppm": matching_ppm, "evaluated_pairs": len(partition_ids)},
        "selection": selection_summary,
        "provenance_counts": provenance_counts,
        "skipped_pairs": skipped,
        "metrics": metrics,
    }
    configured = script_args.dense_counterfactual_output_report or suite.get("output", {}).get("report_path")
    output_path = Path(configured) if configured else Path(global_args.encoder_weights).with_suffix(".dense_precursor_counterfactual.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"skipped_pairs": skipped, "macro": metrics["macro"], "pooled": metrics["pooled"]}, indent=2, sort_keys=True))
    print(f"Wrote dense precursor counterfactual report: {output_path}")
if __name__ == "__main__":
    main()
