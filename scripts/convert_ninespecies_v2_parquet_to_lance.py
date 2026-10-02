import argparse
from pathlib import Path

import lance
import pyarrow.dataset as ds


def _species_from_path(path: Path) -> str:
    return path.stem


def _write_dataset(dataset, out_path: Path, *, overwrite: bool):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "overwrite" if overwrite else "create"
    lance.write_dataset(dataset, str(out_path), mode=mode)


def build_species_datasets(parquet_dir: Path, output_root: Path, *, overwrite: bool):
    parquet_files = sorted(parquet_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {parquet_dir}")

    species = sorted({_species_from_path(p) for p in parquet_files})
    for sp in species:
        out_path = output_root / f"{sp}.lance"
        dataset = ds.dataset(str(parquet_dir / f"{sp}.parquet"), format="parquet")
        _write_dataset(dataset, out_path, overwrite=overwrite)
        print(f"Wrote {sp} -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert 9_species_V2 parquet to lance splits.")
    parser.add_argument(
        "--parquet-dir",
        default="/path/to/data/9_species_V2/parquet/processed",
        help="Directory containing per-species parquet files.",
    )
    parser.add_argument(
        "--output-root",
        default="/path/to/data/9_species_V2/lance_species",
        help="Output root for per-species lance datasets.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing lance outputs.",
    )
    args = parser.parse_args()

    parquet_dir = Path(args.parquet_dir)
    output_root = Path(args.output_root)

    build_species_datasets(parquet_dir, output_root, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
