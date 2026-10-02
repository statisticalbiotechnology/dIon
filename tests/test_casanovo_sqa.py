from pathlib import Path
from zipfile import ZipFile

import numpy as np

from scripts.data_prep.casanovo_sqa.prepare_casanovo_sqa import (
    load_config,
    mapping_records,
    raw_candidates,
    choose_archive_member,
    extract_raw_archive,
    scan_number,
    sha256_file,
    verify_checksum,
)
from scripts.data_prep.casanovo_sqa.balance_casanovo_sqa import (
    balanced_indices,
    supported_charge_indices,
)


CONFIG = Path(
    "scripts/data_prep/casanovo_sqa/"
    "casanovo_sqa_config.json"
)


def test_mapping_is_explicit_and_split_disjoint():
    records = mapping_records(load_config(CONFIG))

    assert len(records) == 19
    assert len({record["msv"] for record in records}) == 19
    assert len({record["pxd"] for record in records}) == 19
    assert {record["split"] for record in records} == {"train", "val", "test"}


def test_raw_candidates_are_sorted_and_raw_only():
    candidates = raw_candidates(
        [
            {"fileName": "z.raw", "fileSizeBytes": 3},
            {"fileName": "notes.txt", "fileSizeBytes": 9},
            {"fileName": "a.RAW", "fileSizeBytes": 1},
        ]
    )

    assert [item["file_name"] for item in candidates] == ["a.RAW", "z.raw"]
    assert candidates[0]["file_size"] == 1


def test_raw_archive_is_fallback_when_no_direct_raw_exists():
    candidates = raw_candidates([
        {"fileName": "run.raw.zip", "fileSizeBytes": 10},
        {"fileName": "notes.txt", "fileSizeBytes": 9},
    ])

    assert len(candidates) == 1
    assert candidates[0]["source_kind"] == "raw_archive"


def test_archive_member_selection_is_deterministic(tmp_path):
    archive = tmp_path / "bundle.raw.zip"
    out = tmp_path / "out"
    out.mkdir()
    with ZipFile(archive, "w") as handle:
        handle.writestr("z.raw", b"z")
        handle.writestr("a.raw", b"a")

    item = {"pxd": "PXD000000", "selected_file": archive.name}
    first = choose_archive_member(archive, item, 0)
    second = choose_archive_member(archive, item, 0)
    assert first == second
    extracted = extract_raw_archive(archive, out, first["archive_member"])
    assert extracted.read_bytes() in {b"a", b"z"}


def test_scan_number_accepts_scan_ids_and_integers():
    assert scan_number("controllerType=0 scan=12345") == 12345
    assert scan_number(12345) == 12345
    assert scan_number("no scan") is None


def test_checksum_verification(tmp_path):
    path = tmp_path / "file.raw"
    path.write_bytes(b"reproducible")
    checksum = sha256_file(path)

    verify_checksum(path, {"checksum": checksum})
    try:
        verify_checksum(path, {"checksum": "0" * 64})
    except ValueError as error:
        assert "Checksum mismatch" in str(error)
    else:
        raise AssertionError("Expected checksum mismatch")



def test_balanced_indices_are_deterministic_and_retain_the_minority_class():
    labels = np.array([0, 0, 0, 0, 0, 1, 1, 1])

    selected, summary = balanced_indices(labels, seed=17)
    selected_again, summary_again = balanced_indices(labels, seed=17)

    assert np.array_equal(selected, selected_again)
    assert summary == summary_again
    assert summary["output_negative"] == summary["output_positive"] == 3
    assert labels[selected].tolist().count(0) == 3
    assert labels[selected].tolist().count(1) == 3
    assert set(np.flatnonzero(labels == 1)).issubset(set(selected.tolist()))


def test_supported_charge_indices_exclude_zero_high_and_fractional_values():
    charges = np.array([0, 1, 2, 10, 11, np.nan, 2.5])

    assert supported_charge_indices(charges, 1, 10).tolist() == [1, 2, 3]
