#!/usr/bin/env python3
"""Prepare the 19-run Casanovo Foundation spectrum-quality benchmark.

The paper names 19 MSV accessions but does not name the selected RAW file
within each mapped PXD. This pipeline records that ambiguity explicitly: it
selects one RAW uniformly from each PXD using a recorded seed, converts with
ThermoRawFileParser, searches all converted files jointly with Sage, and
materializes run-disjoint Parquet splits.

Stages are resumable and refuse to overwrite an existing output. RAW files
are deleted only after a successful, validated conversion unless
--keep-raw is supplied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from tqdm import tqdm


DEFAULT_CONFIG = Path(
    "scripts/data_prep/casanovo_sqa/casanovo_sqa_config.json"
)
DEFAULT_OUTPUT = Path("casanovo_foundation_sqa")
SPLITS = ("train", "val", "test")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_file(path: Path, algorithm: str, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_fingerprint(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def atomic_json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def run_command(command: list[str], log_path: Path, *, cwd: Path | None = None) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("wb") as log:
        process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, cwd=cwd)
    if process.returncode:
        raise RuntimeError(
            f"Command failed with exit code {process.returncode}: {' '.join(command)}; "
            f"see {log_path}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pridepy", help="Override the pridepy executable path")
    parser.add_argument("--converter", help="Override the ThermoRawFileParser path")
    parser.add_argument("--sage", help="Override the Sage executable path")
    parser.add_argument("--sage-config", help="Override the Sage configuration path")
    parser.add_argument("--fasta", help="Override the human FASTA path")
    parser.add_argument(
        "--stage",
        choices=("validate", "metadata", "download", "convert", "search", "materialize", "audit", "all"),
        default="all",
    )
    parser.add_argument("--keep-raw", action="store_true")
    parser.add_argument("--refresh-metadata", action="store_true")
    parser.add_argument("--force-search", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        config = json.load(handle)
    config["_config_path"] = str(path.resolve())
    config["_config_sha256"] = sha256_file(path)
    return config


def mapping_records(config: dict[str, Any]) -> list[dict[str, str]]:
    mapping = config["msv_pxd_split_mapping"]
    records = []
    seen_msv: set[str] = set()
    seen_pxd: set[str] = set()
    for split in SPLITS:
        for msv, pxd in mapping[split].items():
            if msv in seen_msv:
                raise ValueError(f"MSV occurs in multiple splits: {msv}")
            if pxd in seen_pxd:
                raise ValueError(f"PXD occurs in multiple splits: {pxd}")
            seen_msv.add(msv)
            seen_pxd.add(pxd)
            records.append({"split": split, "msv": msv, "pxd": pxd})
    if len(records) != 19:
        raise ValueError(f"Expected 19 explicit MSV/PXD records, found {len(records)}")
    return records


def resolve_config_path(value: str, *, executable: bool = False) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(str(value)))
    path = Path(expanded)
    if not path.is_absolute():
        path = Path.cwd() / path
    if executable and not path.exists():
        discovered = shutil.which(expanded)
        if discovered:
            path = Path(discovered)
    return path.resolve()


def apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    overrides = {
        "pridepy": args.pridepy,
        "converter": args.converter,
        "sage": args.sage,
        "sage_config": args.sage_config,
        "fasta": args.fasta,
    }
    for key, value in overrides.items():
        if value is not None:
            config["paths"][key] = value


def validate_static_inputs(config: dict[str, Any]) -> list[dict[str, str]]:
    records = mapping_records(config)
    paths = config["paths"]
    resolved = {
        key: resolve_config_path(value, executable=key in {"pridepy", "converter", "sage"})
        for key, value in paths.items()
    }
    for key, path in resolved.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Configured {key} does not exist: {path}. "
                "Edit the config or pass the corresponding command-line override."
            )
    for key in ("pridepy", "converter", "sage"):
        path = resolved[key]
        if not os.access(path, os.X_OK):
            raise PermissionError(f"Configured {key} is not executable: {path}")
    config["paths"] = {key: str(path) for key, path in resolved.items()}
    if not 0 <= int(config["selection_seed"]):
        raise ValueError("selection_seed must be non-negative")
    missing_modules = []
    for module in ("numpy", "pandas", "pyarrow", "pyteomics", "psims", "tqdm"):
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)
    if missing_modules:
        raise ImportError(
            "Missing Python dependencies for this pipeline: " + ", ".join(missing_modules)
        )
    return records


def tool_fingerprint(path: Path, version_args: tuple[str, ...] = ("--version",)) -> dict[str, Any]:
    result = subprocess.run(
        [str(path), *version_args], capture_output=True, text=True, check=False
    )
    return {
        **file_fingerprint(path),
        "version_exit_code": result.returncode,
        "version_stdout": result.stdout.strip(),
        "version_stderr": result.stderr.strip(),
    }


def paths_for(root: Path) -> dict[str, Path]:
    return {
        "metadata": root / "metadata",
        "raw": root / "raw",
        "mzml": root / "mzml",
        "logs": root / "logs",
        "sage": root / "sage_joint",
        "manifest": root / "manifest.json",
        "selection": root / "selection_manifest.json",
    }


def manifest_resume_compatible(
    manifest: dict[str, Any], config: dict[str, Any], records: list[dict[str, str]]
) -> bool:
    # Permit only the fallback-policy extension when resuming an older partial run.
    if manifest.get("version") != config.get("version"):
        return False
    if manifest.get("selection_seed") != config.get("selection_seed"):
        return False
    if manifest.get("msv_pxd_split_mapping") != records:
        return False
    if manifest.get("selection_policy") != config.get("selection_policy"):
        return False
    conversion = config["conversion"]
    expected_flags = {
        f"-f={conversion['format']}",
        f"-m={conversion['metadata']}",
        f"-l={conversion['logging']}",
    }
    for entry in manifest.get("conversion", {}).values():
        command = set(entry.get("command") or [])
        if command and not expected_flags.issubset(command):
            return False
    for key in ("sage_config", "fasta", "pridepy", "converter", "sage"):
        previous = manifest.get("input_artifacts", {}).get(key, {}).get("sha256")
        current_path = Path(config["paths"][key])
        if previous and current_path.exists() and sha256_file(current_path) != previous:
            return False
    return True


def load_or_initialize_manifest(
    root: Path, config: dict[str, Any], records: list[dict[str, str]]
) -> dict[str, Any]:
    paths = paths_for(root)
    if paths["manifest"].exists():
        manifest = json.loads(paths["manifest"].read_text())
        if manifest.get("config_sha256") != config["_config_sha256"]:
            if not manifest_resume_compatible(manifest, config, records):
                raise ValueError("Existing manifest was created with a different configuration")
            manifest["config_migration"] = {
                "from_config_sha256": manifest.get("config_sha256"),
                "to_config_sha256": config["_config_sha256"],
                "reason": "Added conversion fallback policy while resuming a partial run",
                "migrated_at_utc": utc_now(),
            }
            manifest["config_sha256"] = config["_config_sha256"]
            manifest["config_path"] = config["_config_path"]
            atomic_json_write(paths["manifest"], manifest)
        if manifest.get("msv_pxd_split_mapping") != records:
            raise ValueError("Existing manifest has a different MSV/PXD mapping")
        return manifest

    manifest = {
        "name": config["name"],
        "version": config["version"],
        "created_at_utc": utc_now(),
        "config_path": config["_config_path"],
        "config_sha256": config["_config_sha256"],
        "selection_seed": config["selection_seed"],
        "msv_pxd_split_mapping": records,
        "documented_run_count": config["selection_policy"]["documented_run_count"],
        "explicit_msv_count": config["selection_policy"]["explicit_msv_count"],
        "run_count_note": config["selection_policy"]["run_count_note"],
        "selection_policy": config["selection_policy"],
        "input_artifacts": {
            "config": file_fingerprint(Path(config["_config_path"])),
            "sage_config": file_fingerprint(Path(config["paths"]["sage_config"])),
            "fasta": file_fingerprint(Path(config["paths"]["fasta"])),
            "pridepy": tool_fingerprint(Path(config["paths"]["pridepy"]), ("--help",)),
            "converter": tool_fingerprint(Path(config["paths"]["converter"])),
            "sage": tool_fingerprint(Path(config["paths"]["sage"])),
        },
        "selected_files": {},
        "download": {},
        "conversion": {},
        "search": {},
        "materialization": {},
    }
    root.mkdir(parents=True, exist_ok=True)
    atomic_json_write(paths["manifest"], manifest)
    return manifest


def metadata_for_pxd(
    *,
    pxd: str,
    config: dict[str, Any],
    root: Path,
    refresh: bool,
) -> list[dict[str, Any]]:
    paths = paths_for(root)
    output = paths["metadata"] / f"{pxd}.json"
    if output.exists() and not refresh:
        return json.loads(output.read_text())

    paths["metadata"].mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=paths["metadata"], prefix=f".{pxd}.", suffix=".json", delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
    command = [
        config["paths"]["pridepy"],
        "stream-files-metadata",
        "-a",
        pxd,
        "-o",
        str(temporary_path),
    ]
    try:
        run_command(command, paths["logs"] / f"metadata_{pxd}.log")
        data = json.loads(temporary_path.read_text())
        atomic_json_write(output, data)
        return data
    finally:
        temporary_path.unlink(missing_ok=True)


def raw_candidates(metadata: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    direct = []
    archives = []
    for entry in metadata:
        name = str(entry.get("fileName", ""))
        lower_name = name.lower()
        if lower_name.endswith(".raw"):
            kind = "raw"
        elif lower_name.endswith(".raw.zip"):
            kind = "raw_archive"
        else:
            continue
        candidate = {
            "file_name": name,
            "file_size": first_int(entry, "fileSize", "fileSizeBytes", "size", "file_size"),
            "checksum": first_value(entry, "checksum", "fileChecksum", "sha256", "md5"),
            "metadata_accession": first_value(entry, "accession", "fileAccession"),
            "source_kind": kind,
        }
        (direct if kind == "raw" else archives).append(candidate)
    # Prefer native RAW files. Some PRIDE projects expose only one-file RAW
    # archives, which are the fallback rather than an additional candidate pool.
    candidates = direct or archives
    return sorted(candidates, key=lambda item: item["file_name"])


def first_value(entry: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if entry.get(key) not in (None, ""):
            return entry[key]
    return None


def first_int(entry: dict[str, Any], *keys: str) -> int | None:
    value = first_value(entry, *keys)
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def selected_raw_path(root: Path, item: dict[str, Any]) -> Path:
    raw_dir = paths_for(root)["raw"] / item["pxd"]
    if item.get("source_kind") == "raw_archive":
        member = item.get("archive_member")
        if not member:
            raise ValueError(f"Archive member has not been selected for {item['pxd']}")
        return raw_dir / Path(member).name
    return raw_dir / item["selected_file"]


def raw_archive_members(archive_path: Path) -> list[str]:
    with zipfile.ZipFile(archive_path) as archive:
        return sorted(
            member
            for member in archive.namelist()
            if not member.endswith("/") and member.lower().endswith(".raw")
        )


def choose_archive_member(archive_path: Path, item: dict[str, Any], seed: int) -> dict[str, Any]:
    members = raw_archive_members(archive_path)
    if not members:
        raise ValueError(f"Archive contains no RAW files: {archive_path}")
    derived_seed = f"{seed}:{item['pxd']}:{item['selected_file']}"
    choice_index = random.Random(derived_seed).randrange(len(members))
    return {
        "archive_member": members[choice_index],
        "archive_member_index": choice_index,
        "archive_member_count": len(members),
    }


def extract_raw_archive(archive_path: Path, raw_dir: Path, member: str) -> Path:
    members = raw_archive_members(archive_path)
    if member not in members:
        raise ValueError(f"Selected RAW is absent from archive {archive_path.name}: {member}")
    target = raw_dir / Path(member).name
    if target.exists():
        return target
    raw_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive, archive.open(member) as source, tempfile.NamedTemporaryFile(
        dir=raw_dir, prefix=f".{target.name}.", suffix=".incomplete", delete=False
    ) as temporary:
        shutil.copyfileobj(source, temporary)
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, target)
    return target


def select_files(
    config: dict[str, Any], root: Path, records: list[dict[str, str]], manifest: dict[str, Any], refresh: bool
) -> None:
    if manifest["selected_files"] and not refresh:
        changed = False
        for item in manifest["selected_files"].values():
            if "source_kind" not in item:
                candidate = next(
                    candidate
                    for candidate in item["candidate_files"]
                    if candidate["file_name"] == item["selected_file"]
                )
                item["source_kind"] = candidate.get("source_kind", "raw")
                changed = True
        if changed:
            atomic_json_write(paths_for(root)["manifest"], manifest)
            atomic_json_write(
                paths_for(root)["selection"],
                {
                    "selection_seed": config["selection_seed"],
                    "policy": config["selection_policy"],
                    "records": manifest["selected_files"],
                },
            )
        return
    rng = random.Random(int(config["selection_seed"]))
    selected: dict[str, Any] = {}
    for record in tqdm(records, desc="Selecting RAW files", unit="PXD"):
        candidates = raw_candidates(metadata_for_pxd(pxd=record["pxd"], config=config, root=root, refresh=refresh))
        if not candidates:
            raise ValueError(f"No RAW candidates found for {record['pxd']}")
        choice_index = rng.randrange(len(candidates))
        choice = candidates[choice_index]
        metadata_path = paths_for(root)["metadata"] / f"{record['pxd']}.json"
        selected[record["pxd"]] = {
            **record,
            "metadata_path": str(metadata_path),
            "metadata_sha256": sha256_file(metadata_path),
            "candidate_count": len(candidates),
            "candidate_files": candidates,
            "selected_index": choice_index,
            "selected_file": choice["file_name"],
            "source_kind": choice.get("source_kind", "raw"),
        }
    manifest["selected_files"] = selected
    atomic_json_write(
        paths_for(root)["selection"],
        {
            "selection_seed": config["selection_seed"],
            "policy": config["selection_policy"],
            "records": selected,
        },
    )
    atomic_json_write(paths_for(root)["manifest"], manifest)


def selected_records(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    selected = list(manifest["selected_files"].values())
    if len(selected) != 19:
        raise ValueError(f"Expected 19 selected files, found {len(selected)}")
    return sorted(selected, key=lambda item: (SPLITS.index(item["split"]), item["msv"]))


def expected_mzml_path(root: Path, item: dict[str, Any]) -> Path:
    # Archive candidates are renamed to the selected RAW member before conversion.
    selected_name = item.get("archive_member", item["selected_file"])
    stem = Path(selected_name).stem
    return paths_for(root)["mzml"] / item["pxd"] / f"{item['pxd']}__{stem}.mzML"


def validate_existing_mzml(path: Path) -> dict[str, int]:
    if not path.exists() or path.stat().st_size == 0:
        raise ValueError(f"Missing or empty mzML: {path}")
    namespace = "{http://psi.hupo.org/ms/mzml}"
    counts = Counter()
    for _, element in ET.iterparse(path, events=("end",)):
        if element.tag == namespace + "spectrum":
            counts["spectra"] += 1
            levels = [
                param.attrib.get("value")
                for param in element.findall(namespace + "cvParam")
                if param.attrib.get("accession") == "MS:1000511"
            ]
            counts[f"ms{levels[0] if levels else 'unknown'}"] += 1
            element.clear()
    if not counts["spectra"]:
        raise ValueError(f"mzML contains no spectra: {path}")
    return dict(counts)


def persist_selection(root: Path, config: dict[str, Any], manifest: dict[str, Any]) -> None:
    atomic_json_write(
        paths_for(root)["selection"],
        {
            "selection_seed": config["selection_seed"],
            "policy": config["selection_policy"],
            "records": manifest["selected_files"],
        },
    )
    atomic_json_write(paths_for(root)["manifest"], manifest)


def candidate_for(item: dict[str, Any], file_name: str) -> dict[str, Any]:
    return next(candidate for candidate in item["candidate_files"] if candidate["file_name"] == file_name)


def download_candidate(
    config: dict[str, Any],
    root: Path,
    item: dict[str, Any],
    candidate: dict[str, Any],
    *,
    archive_member: str | None = None,
    log_suffix: str = "",
) -> tuple[Path, dict[str, Any]]:
    paths = paths_for(root)
    pxd = item["pxd"]
    file_name = candidate["file_name"]
    source_kind = candidate.get("source_kind", "raw")
    raw_dir = paths["raw"] / pxd
    raw_dir.mkdir(parents=True, exist_ok=True)
    source_path = raw_dir / file_name
    expected_size = candidate.get("file_size")

    if source_kind == "raw_archive":
        archive_member = archive_member or (
            item.get("archive_member") if item.get("selected_file") == file_name else None
        )
        if archive_member:
            extracted_path = raw_dir / Path(archive_member).name
            if extracted_path.exists():
                return extracted_path, {
                    "source_path": str(source_path) if source_path.exists() else None,
                    "archive_member": archive_member,
                    "archive_member_index": item.get("archive_member_index"),
                    "archive_member_count": item.get("archive_member_count"),
                    "source_kind": source_kind,
                }
    else:
        extracted_path = source_path

    if not source_path.exists():
        staging = paths["raw"] / ".download_staging" / pxd
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True, exist_ok=True)
        command = [
            config["paths"]["pridepy"],
            "download-file-by-name",
            "-a",
            pxd,
            "-f",
            file_name,
            "-o",
            str(staging),
            "-p",
            config["download_protocol"],
        ]
        log_name = f"download_{pxd}{log_suffix}.log"
        run_command(command, paths["logs"] / log_name)
        downloaded = staging / file_name
        if not downloaded.exists():
            matches = list(staging.rglob(file_name))
            if len(matches) == 1:
                downloaded = matches[0]
        if not downloaded.exists():
            raise FileNotFoundError(f"PRIDE download did not produce {file_name}")
        if expected_size is not None and downloaded.stat().st_size != expected_size:
            raise ValueError(
                f"Downloaded size mismatch for {pxd}/{file_name}: "
                f"{downloaded.stat().st_size} != {expected_size}"
            )
        verify_checksum(downloaded, candidate)
        os.replace(downloaded, source_path)
        shutil.rmtree(staging, ignore_errors=True)

    if expected_size is not None and source_path.stat().st_size != expected_size:
        raise ValueError(
            f"Existing source size mismatch for {pxd}/{file_name}: "
            f"{source_path.stat().st_size} != {expected_size}"
        )
    verify_checksum(source_path, candidate)

    archive_info: dict[str, Any] = {}
    if source_kind == "raw_archive":
        if archive_member is None:
            archive_info = choose_archive_member(
                source_path,
                {**item, "selected_file": file_name},
                int(config["selection_seed"]),
            )
            archive_member = archive_info["archive_member"]
        extracted_path = extract_raw_archive(source_path, raw_dir, archive_member)
        source_path.unlink()
    return extracted_path, {
        "source_path": str(source_path) if source_path.exists() else None,
        **archive_info,
        "archive_member": archive_member,
        "source_kind": source_kind,
    }


def record_download(
    root: Path,
    manifest: dict[str, Any],
    item: dict[str, Any],
    raw_path: Path,
    provenance: dict[str, Any],
    status: str,
) -> None:
    candidate = candidate_for(item, item["selected_file"])
    manifest["download"][item["pxd"]] = {
        "status": status,
        "path": str(raw_path),
        "source_path": provenance.get("source_path"),
        "size_bytes": raw_path.stat().st_size,
        "sha256": sha256_file(raw_path),
        "checksum": candidate.get("checksum"),
        "source_kind": item.get("source_kind", "raw"),
        **{
            key: provenance[key]
            for key in ("archive_member", "archive_member_index", "archive_member_count")
            if provenance.get(key) is not None
        },
    }
    atomic_json_write(paths_for(root)["manifest"], manifest)


def download_selected(config: dict[str, Any], root: Path, manifest: dict[str, Any]) -> None:
    for item in tqdm(selected_records(manifest), desc="Downloading selected RAWs", unit="file"):
        # Successful conversions may have removed their RAW input. Reuse the
        # validated mzML instead of downloading that RAW again on resume.
        pxd = item["pxd"]
        conversion = manifest.get("conversion", {}).get(pxd)
        if conversion:
            mzml_path = expected_mzml_path(root, item)
            if mzml_path.exists() and mzml_path.stat().st_size > 0:
                continue
        candidate = candidate_for(item, item["selected_file"])
        raw_path, provenance = download_candidate(
            config,
            root,
            item,
            candidate,
            archive_member=item.get("archive_member"),
        )
        if candidate.get("source_kind") == "raw_archive":
            item.update(provenance)
            item["source_kind"] = "raw_archive"
            persist_selection(root, config, manifest)
        record_download(
            root,
            manifest,
            item,
            raw_path,
            provenance,
            "reused" if raw_path.exists() else "downloaded",
        )


def verify_checksum(path: Path, item: dict[str, Any]) -> None:
    expected = str(item.get("checksum") or "").strip().lower()
    if not expected or not re.fullmatch(r"[0-9a-f]+", expected):
        return
    algorithm_by_length = {32: "md5", 40: "sha1", 64: "sha256"}
    algorithm = algorithm_by_length.get(len(expected))
    if algorithm is None:
        return
    actual = hash_file(path, algorithm)
    if actual != expected:
        raise ValueError(
            f"Checksum mismatch for {path.name}: {algorithm} {actual} != {expected}"
        )


def convert_selected(config: dict[str, Any], root: Path, manifest: dict[str, Any], keep_raw: bool) -> None:
    paths = paths_for(root)
    conversion_config = config["conversion"]
    allow_fallback = bool(conversion_config.get("fallback_on_error", True))
    for item in tqdm(selected_records(manifest), desc="Converting RAW files", unit="file"):
        pxd = item["pxd"]
        attempts = list(item.get("conversion_attempts", []))
        attempted_names = {attempt["file_name"] for attempt in attempts}
        current_name = item["selected_file"]
        candidates = [candidate_for(item, current_name)]
        if allow_fallback:
            candidates.extend(
                sorted(
                    (
                        candidate
                        for candidate in item["candidate_files"]
                        if candidate["file_name"] != current_name
                        and candidate["file_name"] not in attempted_names
                    ),
                    key=lambda candidate: (
                        candidate.get("file_size") is None,
                        candidate.get("file_size") or 0,
                        candidate["file_name"],
                    ),
                )
            )
        if current_name in attempted_names:
            candidates = candidates[1:]

        converted = False
        existing_conversion = manifest.get("conversion", {}).get(pxd)
        existing_mzml = expected_mzml_path(root, item)
        if (
            existing_conversion
            and existing_conversion.get("path")
            and Path(existing_conversion["path"]).resolve() == existing_mzml.resolve()
            and existing_mzml.exists()
            and existing_mzml.stat().st_size > 0
        ):
            continue

        for candidate_number, candidate in enumerate(candidates, start=1):
            file_name = candidate["file_name"]
            candidate_index = next(
                index
                for index, entry in enumerate(item["candidate_files"])
                if entry["file_name"] == file_name
            )
            is_fallback = file_name != current_name
            if is_fallback:
                previous_name = item["selected_file"]
                item["selected_file"] = file_name
                item["selected_index"] = candidate_index
                item["source_kind"] = candidate.get("source_kind", "raw")
                for key in ("archive_member", "archive_member_index", "archive_member_count"):
                    item.pop(key, None)
            else:
                previous_name = item.get("fallback_from")

            try:
                raw_path, provenance = download_candidate(
                    config,
                    root,
                    item,
                    candidate,
                    archive_member=item.get("archive_member") if not is_fallback else None,
                    log_suffix=f"_attempt{candidate_number}" if candidate_number > 1 else "",
                )
                if candidate.get("source_kind") == "raw_archive":
                    item.update(provenance)
                else:
                    item["source_kind"] = "raw"
                if is_fallback:
                    item["fallback_from"] = previous_name
                    item["fallback_reason"] = "previous RAW failed conversion"
                    persist_selection(root, config, manifest)
                    record_download(
                        root,
                        manifest,
                        item,
                        raw_path,
                        provenance,
                        "downloaded_for_conversion_fallback",
                    )

                mzml_path = expected_mzml_path(root, item)
                if mzml_path.exists():
                    stats = validate_existing_mzml(mzml_path)
                    manifest["conversion"][pxd] = {
                        "status": "reused",
                        "path": str(mzml_path),
                        "command": None,
                        "attempts": attempts,
                        **stats,
                    }
                    atomic_json_write(paths["manifest"], manifest)
                    converted = True
                    break
                if not raw_path.exists():
                    raise FileNotFoundError(f"RAW missing and mzML absent for {pxd}: {raw_path}")

                outdir = paths["mzml"] / pxd
                outdir.mkdir(parents=True, exist_ok=True)
                workdir = outdir / ".conversion_incomplete"
                if workdir.exists():
                    shutil.rmtree(workdir)
                workdir.mkdir(parents=True, exist_ok=True)
                command = [
                    config["paths"]["converter"],
                    f"-i={raw_path}",
                    f"-o={workdir}",
                    f"-f={conversion_config['format']}",
                    f"-m={conversion_config['metadata']}",
                    f"-l={conversion_config['logging']}",
                ]
                try:
                    run_command(
                        command,
                        paths["logs"]
                        / f"convert_{pxd}{f'_attempt{candidate_number}' if candidate_number > 1 else ''}.log",
                    )
                    generated = workdir / f"{Path(raw_path).stem}.mzML"
                    if not generated.exists():
                        raise FileNotFoundError(
                            f"Converter did not produce expected mzML: {generated}"
                        )
                    generated.rename(mzml_path)
                    stats = validate_existing_mzml(mzml_path)
                finally:
                    shutil.rmtree(workdir, ignore_errors=True)
            except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
                attempts.append(
                    {
                        "file_name": file_name,
                        "error": str(error),
                        "attempt": len(attempts) + 1,
                    }
                )
                item["conversion_attempts"] = attempts
                persist_selection(root, config, manifest)
                if not allow_fallback or candidate_number == len(candidates):
                    raise RuntimeError(
                        f"All conversion candidates failed for {pxd}; "
                        f"attempts: {[attempt['file_name'] for attempt in attempts]}"
                    ) from error
                continue

            raw_hash = sha256_file(raw_path)
            if not keep_raw and not conversion_config["retain_raw_after_successful_conversion"]:
                raw_path.unlink()
            item["conversion_attempts"] = attempts
            manifest["conversion"][pxd] = {
                "status": "converted",
                "path": str(mzml_path),
                "raw_sha256": raw_hash,
                "command": command,
                "attempts": attempts,
                **stats,
            }
            atomic_json_write(paths["manifest"], manifest)
            converted = True
            break

        if not converted:
            raise RuntimeError(f"No conversion candidate available for {pxd}")


def run_joint_search(config: dict[str, Any], root: Path, manifest: dict[str, Any], force: bool) -> None:
    paths = paths_for(root)
    sage_config = json.loads(Path(config["paths"]["sage_config"]).read_text())
    sage_config.setdefault("database", {})["fasta"] = config["paths"]["fasta"]
    runtime_config = paths["sage"] / "sage_runtime_config.json"
    atomic_json_write(runtime_config, sage_config)
    items = selected_records(manifest)
    mzmls = [expected_mzml_path(root, item) for item in items]
    missing = [str(path) for path in mzmls if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Cannot jointly search; missing mzML files: {missing}")
    results_json = paths["sage"] / "results.json"
    results_tsv = paths["sage"] / "results.sage.tsv"
    if results_json.exists() and results_tsv.exists() and not force:
        payload = json.loads(results_json.read_text())
        recorded = [str(Path(path).resolve()) for path in payload.get("mzml_paths", [])]
        expected = [str(path.resolve()) for path in mzmls]
        if recorded == expected:
            manifest["search"] = {
                "status": "reused",
                "results_json": str(results_json),
                "results_tsv": str(results_tsv),
                "sage_reported_version": payload.get("version"),
                "command": None,
                "input_files": expected,
            }
            atomic_json_write(paths["manifest"], manifest)
            return
        raise ValueError("Existing Sage results contain a different input file list; use --force-search")

    if paths["sage"].exists() and force:
        shutil.rmtree(paths["sage"])
    paths["sage"].mkdir(parents=True, exist_ok=True)
    command = [config["paths"]["sage"], str(runtime_config), "-o", str(paths["sage"])]
    if config["search"].get("disable_telemetry", False):
        command.append("--disable-telemetry-i-dont-want-to-improve-sage")
    command.extend(str(path) for path in mzmls)
    run_command(command, paths["logs"] / "sage_joint.log")
    if not results_json.exists() or not results_tsv.exists():
        raise FileNotFoundError("Sage exited successfully but did not write both result files")
    payload = json.loads(results_json.read_text())
    version_result = subprocess.run(
        [config["paths"]["sage"], "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    manifest["search"] = {
        "status": "searched",
        "results_json": str(results_json),
        "results_tsv": str(results_tsv),
        "sage_reported_version": payload.get("version"),
        "sage_command_version": version_result.stdout.strip(),
        "sage_command_version_stderr": version_result.stderr.strip(),
        "command": command,
        "input_files": [str(path.resolve()) for path in mzmls],
        "joint_search": True,
    }
    atomic_json_write(paths["manifest"], manifest)


def scan_number(value: Any) -> int | None:
    match = re.search(r"scan=(\d+)", str(value))
    if match:
        return int(match.group(1))
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_sage_table(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    import pandas as pd

    frame = pd.read_csv(path, sep="\t")
    required = {"filename", "scannr", "peptide_q"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Sage TSV missing columns: {sorted(missing)}")
    frame["source_file"] = frame["filename"].astype(str).map(lambda value: Path(value).name)
    frame["scan_num"] = frame["scannr"].map(scan_number)
    frame["peptide_q_num"] = pd.to_numeric(frame["peptide_q"], errors="coerce").fillna(1.0)
    if "label" not in frame.columns:
        raise ValueError("Sage TSV has no label column; target/decoy status is required")
    frame["is_target"] = pd.to_numeric(frame["label"], errors="coerce").fillna(0) > 0
    frame = frame.loc[frame["scan_num"].notna()].copy()
    frame["scan_num"] = frame["scan_num"].astype(int)
    frame = frame.sort_values(
        ["source_file", "scan_num", "is_target", "peptide_q_num"],
        ascending=[True, True, False, True],
    )
    best: dict[tuple[str, int], dict[str, Any]] = {}
    for _, row in frame.drop_duplicates(["source_file", "scan_num"]).iterrows():
        record = row.to_dict()
        key = (str(record["source_file"]), int(record["scan_num"]))
        best[key] = record
    return best


def parse_mzml_records(path: Path, source: dict[str, Any], sage: dict[tuple[str, int], dict[str, Any]]) -> list[dict[str, Any]]:
    from pyteomics import mzml

    records = []
    with mzml.read(str(path), iterative=True, use_index=False, huge_tree=True) as reader:
        for spectrum in reader:
            if spectrum.get("ms level") != 2:
                continue
            scan = scan_number(spectrum.get("id"))
            if scan is None:
                continue
            source_file = path.name
            psm = sage.get((source_file, scan))
            q_value = float(psm["peptide_q_num"]) if psm is not None else 1.0
            is_target = bool(psm["is_target"]) if psm is not None else False
            label = int(is_target and q_value < 0.01)
            precursor_mz = None
            precursor_charge = None
            precursors = spectrum.get("precursorList", {}).get("precursor", [])
            if precursors:
                ions = precursors[0].get("selectedIonList", {}).get("selectedIon", [])
                if ions:
                    precursor_mz = ions[0].get("selected ion m/z")
                    precursor_charge = ions[0].get("charge state")
                    if precursor_mz is not None:
                        precursor_mz = float(precursor_mz)
                    if precursor_charge is not None:
                        precursor_charge = int(precursor_charge)
            retention_time = spectrum.get("scanList", {}).get("scan", [{}])[0].get(
                "scan start time"
            )
            if retention_time is not None:
                retention_time = float(retention_time)
            mz_array = spectrum.get("m/z array", [])
            intensity_array = spectrum.get("intensity array", [])
            records.append(
                {
                    "scan_num": int(scan),
                    "retention_time": retention_time,
                    "mz_array": mz_array.tolist() if hasattr(mz_array, "tolist") else list(mz_array),
                    "intensity_array": intensity_array.tolist() if hasattr(intensity_array, "tolist") else list(intensity_array),
                    "precursor_mz": precursor_mz,
                    "precursor_charge": precursor_charge,
                    "source_file": source_file,
                    "project_accession": source["pxd"],
                    "msv_accession": source["msv"],
                    "split": source["split"],
                    "label": label,
                    "quality": label,
                    "sage_matched": psm is not None,
                    "sage_target": is_target,
                    "peptide_q": q_value,
                    "peptide": None if psm is None else psm.get("peptide"),
                    "proteins": None if psm is None else psm.get("proteins"),
                    "spectrum_q": None if psm is None else psm.get("spectrum_q"),
                }
            )
    return records


def materialize(config: dict[str, Any], root: Path, manifest: dict[str, Any]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    paths = paths_for(root)
    if manifest.get("materialization", {}).get("status") == "complete":
        if all((root / f"{split}.parquet").exists() for split in SPLITS):
            return
        raise ValueError("Manifest says materialization is complete but a split file is missing")
    sage = parse_sage_table(paths["sage"] / "results.sage.tsv")
    schema = pa.schema(
        [
            ("scan_num", pa.int64()),
            ("retention_time", pa.float64()),
            ("mz_array", pa.list_(pa.float64())),
            ("intensity_array", pa.list_(pa.float64())),
            ("precursor_mz", pa.float64()),
            ("precursor_charge", pa.int64()),
            ("source_file", pa.string()),
            ("project_accession", pa.string()),
            ("msv_accession", pa.string()),
            ("split", pa.string()),
            ("label", pa.int8()),
            ("quality", pa.int8()),
            ("sage_matched", pa.bool_()),
            ("sage_target", pa.bool_()),
            ("peptide_q", pa.float64()),
            ("peptide", pa.string()),
            ("proteins", pa.string()),
            ("spectrum_q", pa.float64()),
        ]
    )
    writers: dict[str, pq.ParquetWriter] = {}
    temporary_outputs = {split: root / f".{split}.parquet.incomplete" for split in SPLITS}
    summaries: dict[str, Any] = {}
    try:
        for output in temporary_outputs.values():
            output.unlink(missing_ok=True)
        for item in tqdm(selected_records(manifest), desc="Materializing SQA Parquet", unit="file"):
            split = item["split"]
            frame_records = parse_mzml_records(expected_mzml_path(root, item), item, sage)
            if not frame_records:
                raise ValueError(f"No MS2 records parsed from {expected_mzml_path(root, item)}")
            table = pa.Table.from_pylist(frame_records, schema=schema)
            output = temporary_outputs[split]
            if split not in writers:
                writers[split] = pq.ParquetWriter(output, schema, compression="zstd")
                summaries[split] = {"rows": 0, "positive": 0, "negative": 0, "source_files": []}
            writers[split].write_table(table)
            summaries[split]["rows"] += len(frame_records)
            summaries[split]["positive"] += sum(record["label"] for record in frame_records)
            summaries[split]["negative"] += sum(1 - record["label"] for record in frame_records)
            summaries[split]["source_files"].append(item["selected_file"])
    except Exception:
        for output in temporary_outputs.values():
            output.unlink(missing_ok=True)
        raise
    finally:
        for writer in writers.values():
            writer.close()
    for split in SPLITS:
        final_output = root / f"{split}.parquet"
        if final_output.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {final_output}")
        os.replace(temporary_outputs[split], final_output)
    for split in SPLITS:
        if split not in summaries:
            raise ValueError(f"No materialized records for split: {split}")
        summaries[split]["source_files"] = sorted(summaries[split]["source_files"])
    manifest["materialization"] = {
        "status": "complete",
        "label_definition": config["search"]["label_rule"],
        "sampling": "none; natural class prevalence retained",
        "splits": summaries,
        "parquet_schema": str(schema),
    }
    atomic_json_write(paths["manifest"], manifest)


def audit(config: dict[str, Any], root: Path, manifest: dict[str, Any]) -> None:
    import pyarrow.parquet as pq

    paths = paths_for(root)
    records = selected_records(manifest)
    expected_by_split = {split: {item["pxd"] for item in records if item["split"] == split} for split in SPLITS}
    expected_files_by_split = {
        split: {expected_mzml_path(root, item).name for item in records if item["split"] == split}
        for split in SPLITS
    }
    expected_msv_by_project = {item["pxd"]: item["msv"] for item in records}
    seen_projects: dict[str, str] = {}
    report: dict[str, Any] = {"source_leakage": [], "split_summaries": {}}
    for split in SPLITS:
        path = root / f"{split}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing split output: {path}")
        table = pq.read_table(
            path,
            columns=["project_accession", "source_file", "msv_accession", "split", "label"],
        )
        frame = table.to_pydict()
        projects = set(frame["project_accession"])
        source_files = set(frame["source_file"])
        if source_files != expected_files_by_split[split]:
            raise ValueError(
                f"{split} contains source files {sorted(source_files)}, "
                f"expected {sorted(expected_files_by_split[split])}"
            )
        if set(frame["split"]) != {split}:
            raise ValueError(f"{split} contains rows with an incorrect split value")
        expected_msvs = {expected_msv_by_project[project] for project in projects}
        if set(frame["msv_accession"]) != expected_msvs:
            raise ValueError(f"{split} has inconsistent MSV provenance")
        if projects != expected_by_split[split]:
            raise ValueError(f"{split} contains projects {sorted(projects)}, expected {sorted(expected_by_split[split])}")
        for project in projects:
            previous = seen_projects.setdefault(project, split)
            if previous != split:
                report["source_leakage"].append(project)
        labels = Counter(int(value) for value in frame["label"])
        report["split_summaries"][split] = {
            "rows": len(frame["label"]),
            "positive": labels[1],
            "negative": labels[0],
            "projects": sorted(projects),
        }
    if report["source_leakage"]:
        raise ValueError(f"Project leakage detected: {report['source_leakage']}")
    report["source_leakage_count"] = 0
    report["completed_at_utc"] = utc_now()
    manifest["audit"] = report
    atomic_json_write(paths["manifest"], manifest)
    print(json.dumps(report, indent=2, sort_keys=True))


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    apply_cli_overrides(config, args)
    records = validate_static_inputs(config)
    if args.stage == "validate":
        print(json.dumps({"config": str(args.config.resolve()), "config_sha256": config["_config_sha256"], "records": records}, indent=2))
        return

    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest = load_or_initialize_manifest(root, config, records)
    if args.refresh_metadata:
        for item in tqdm(records, desc="Refreshing PRIDE metadata", unit="PXD"):
            metadata_for_pxd(pxd=item["pxd"], config=config, root=root, refresh=True)
    select_files(config, root, records, manifest, refresh=False)

    stages = {
        "metadata": ("metadata",),
        "download": ("metadata", "download"),
        "convert": ("metadata", "download", "convert"),
        "search": ("metadata", "download", "convert", "search"),
        "materialize": ("metadata", "download", "convert", "search", "materialize"),
        "audit": ("metadata", "download", "convert", "search", "materialize", "audit"),
        "all": ("metadata", "download", "convert", "search", "materialize", "audit"),
    }
    requested = stages[args.stage]
    if "metadata" in requested:
        for item in tqdm(records, desc="Fetching PRIDE metadata", unit="PXD"):
            metadata_for_pxd(pxd=item["pxd"], config=config, root=root, refresh=args.refresh_metadata)
    if "download" in requested:
        download_selected(config, root, manifest)
    if "convert" in requested:
        convert_selected(config, root, manifest, args.keep_raw)
    if "search" in requested:
        run_joint_search(config, root, manifest, args.force_search)
    if "materialize" in requested:
        materialize(config, root, manifest)
    if "audit" in requested:
        audit(config, root, manifest)
    print(f"Wrote/updated Casanovo SQA manifest: {paths_for(root)['manifest']}")


if __name__ == "__main__":
    main()
