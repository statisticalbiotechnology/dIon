"""Test whether null-conditioned 60% local peak views preserve source identity."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
import yaml
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.distractor_augmentation import BatchedStudentDistractorMixAugmentation
from src.wrappers.pretrain_wrappers import _build_dino_augmentation
from src.embed_eval.counterfactual import (
    evaluate_null_local_source_consistency,
    select_counterfactual_pairs,
)
from src.embed_eval.data import PeptideRetrievalDataset, build_embedding_eval_collate
from src.embed_eval.extraction import preserve_eval_mode
from src.embed_eval.loading import load_checkpoint_embedder
from src.parse_args import parse_args_and_config


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


def main() -> None:
    global_args, pretrain_config, _, probing_config = parse_args_and_config()
    suite_key = "null_local_consistency_evaluation"
    if not probing_config or suite_key not in probing_config:
        raise ValueError(f"--probing_config must contain {suite_key!r}.")
    if global_args.embedding_baseline != "model":
        raise ValueError("Null-local consistency evaluation requires model embeddings.")
    if global_args.embedding_readout != "backbone":
        raise ValueError("Null-local consistency evaluation currently requires --embedding_readout backbone.")
    if global_args.counterfactual_distance != "cosine":
        raise ValueError("Null-local consistency evaluation currently requires --counterfactual_distance cosine.")
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
    pair_eligibility = suite["pair_eligibility"]
    mixer = BatchedStudentDistractorMixAugmentation(
        mix_apply_to="all",
        condition_separation_ppm=float(pair_eligibility.get("condition_separation_ppm", 10.0)),
        neutral_mass_separation_ppm=float(pair_eligibility.get("neutral_mass_separation_ppm", 10.0)),
        merge_ppm=float(pair_eligibility.get("merge_ppm", 5.0)),
        intensity_normalization="none",
    )
    augmentation = _build_dino_augmentation(pretrain_config[global_args.pretraining_task], global_args)
    num_global_crops = int(pretrain_config[global_args.pretraining_task]["num_global_crops"])
    num_local_crops = int(pretrain_config[global_args.pretraining_task]["num_local_crops"])
    if num_local_crops < 2:
        raise ValueError("Null-local consistency evaluation requires at least two configured local crops.")
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
                torch.manual_seed(_stable_seed(base_seed, "local-crops", *complete_pair_ids))
                crops = augmentation(
                    anchor["peaks"],
                    (~anchor["padding"]).sum(dim=1),
                    rand_size=False,
                )
            local_one_peaks, local_one_padding = crops[num_global_crops]
            local_two_peaks, local_two_padding = crops[num_global_crops + 1]
            null_mass = torch.zeros_like(anchor["mass"])
            null_charge = torch.zeros_like(anchor["charge"])
            local_one = {
                **anchor,
                "peaks": local_one_peaks,
                "padding": local_one_padding,
                "mass": null_mass,
                "charge": null_charge,
            }
            local_two = {
                **anchor,
                "peaks": local_two_peaks,
                "padding": local_two_padding,
                "mass": null_mass,
                "charge": null_charge,
            }
            clean_a = _embed(embedder, anchor)
            clean_b = _embed(embedder, distractor)
            local_one_value = _embed(embedder, local_one)
            local_two_value = _embed(embedder, local_two)
            clean_a_values.append(clean_a.cpu())
            clean_b_values.append(clean_b.cpu())
            mixed_a_values.append(local_one_value.cpu())
            mixed_b_values.append(local_two_value.cpu())
            partition_ids.extend(
                str(batch["partition_id"][index]) for index in anchor_positions.tolist()
            )
            processed_pairs += anchor_positions.numel()
    if processed_pairs != len(records):
        raise RuntimeError(
            f"Only {processed_pairs}/{len(records)} pairs survived preprocessing; "
            "counterfactual evaluation requires complete A/B pairs."
        )
    metrics = evaluate_null_local_source_consistency(
        torch.cat(clean_a_values), torch.cat(clean_b_values),
        torch.cat(mixed_a_values), torch.cat(mixed_b_values), partition_ids,
    )
    report = {
        "checkpoint_path": global_args.encoder_weights,
        "embedding_readout": global_args.embedding_readout,
        "distance_metric": global_args.counterfactual_distance,
        "representation": "EMA-teacher pooled backbone features",
        "protocol": "null_local_view_consistency",
        "description": (
            "Two independent configured 60% local peak subsets of A are embedded "
            "with the learned null precursor (mass=0, charge=0); each is compared "
            "with full clean conditioned A and a matched different-peptide B control."
        ),
        "metric_definitions": {
            "local_view_1_source_accuracy": "local_view_1(A, NULL) selects clean A over clean B",
            "local_view_2_source_accuracy": "local_view_2(A, NULL) selects clean A over clean B",
            "paired_source_accuracy": "both independent local views select clean A",
            "local_view_1_margin": "cosine(local_view_1(A, NULL), clean_A) - cosine(local_view_1(A, NULL), clean_B)",
            "local_view_2_margin": "cosine(local_view_2(A, NULL), clean_A) - cosine(local_view_2(A, NULL), clean_B)",
        },
        "dataset": {"name": retrieval_config["name"], "selection": dataset.selection.__dict__},
        "local_crop": {
            "selection_mode": pretrain_config[global_args.pretraining_task]["selection_mode"],
            "local_crops_scale": pretrain_config[global_args.pretraining_task]["local_crops_scale"],
            "num_independent_views": 2,
            "precursor_mass": 0.0,
            "precursor_charge": 0,
        },
        "pair_eligibility": pair_eligibility,
        "selection": selection_summary,
        "metrics": metrics,
    }
    configured = global_args.counterfactual_output_report or suite.get("output", {}).get("report_path")
    output_path = Path(configured) if configured else Path(global_args.encoder_weights).with_suffix(".null_local_consistency.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"selection": selection_summary, "macro": metrics["macro"], "pooled": metrics["pooled"]}, indent=2, sort_keys=True))
    print(f"Wrote null-local consistency report: {output_path}")


if __name__ == "__main__":
    main()
