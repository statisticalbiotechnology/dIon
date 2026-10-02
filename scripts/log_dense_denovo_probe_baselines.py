"""Log dense de novo probe reference runs to Weights & Biases.

The references are deliberately separate runs because they have different
scientific status from ordinary DINO training: a random frozen floor, the
current frozen SSL checkpoint, and a supervised in-domain reference whose
training corpus overlaps this development probe.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_baseline(value: str) -> tuple[str, str, Path]:
    try:
        name, role, report_path = value.split(":", 2)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--baseline must be NAME:ROLE:REPORT_JSON"
        ) from exc
    return name, role, Path(report_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Log static dense de novo probe references to W&B."
    )
    parser.add_argument("--entity", default="user")
    parser.add_argument("--project", default="dion-pretraining")
    parser.add_argument(
        "--group",
        default="dense_denovo_probe_references_v1",
        help="W&B group reserved for static probe reference runs.",
    )
    parser.add_argument(
        "--global_steps",
        type=int,
        nargs="+",
        default=[0, 100000],
        help="Static x-axis anchors used to draw each reference in probe charts.",
    )
    parser.add_argument(
        "--baseline",
        action="append",
        type=parse_baseline,
        required=True,
        metavar="NAME:ROLE:REPORT_JSON",
        help=(
            "Reference report and interpretation. Roles: frozen_random_floor, "
            "frozen_ssl_reference, or supervised_leaky_reference."
        ),
    )
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(set(args.global_steps)) != len(args.global_steps) or min(args.global_steps) < 0:
        raise ValueError("--global_steps must be unique non-negative integers.")

    for name, role, report_path in args.baseline:
        report = json.loads(report_path.read_text())
        metrics = report.get("metrics")
        if not isinstance(metrics, dict):
            raise ValueError(f"{report_path} does not contain a metrics object.")
        run_name = f"dense_denovo_probe_{name}"
        metadata = {
            "dense_denovo_probe_reference": True,
            "dense_denovo_probe_reference_role": role,
            "dense_denovo_probe_reference_report": str(report_path.resolve()),
            "encoder_source": report.get("encoder_source"),
            "encoder_source_type": report.get("encoder_source_type"),
            "encoder_precursor_conditioning": report.get("encoder_precursor_conditioning"),
            "scientific_status": (
                "development floor; untrained frozen encoder"
                if role == "frozen_random_floor"
                else "frozen SSL reference checkpoint"
                if role == "frozen_ssl_reference"
                else "supervised in-domain reference; probe data overlap is expected"
            ),
        }
        payload = {
            "probe/dense_denovo_is_baseline": 1,
            "probe/dense_denovo_reference_role": role,
            **metrics,
        }
        if args.dry_run:
            print(json.dumps({"run_name": run_name, "metadata": metadata, "payload": payload}, indent=2))
            continue

        import wandb

        run = wandb.init(
            entity=args.entity,
            project=args.project,
            name=run_name,
            group=args.group,
            job_type="dense_denovo_probe_reference",
            config=metadata,
            reinit="create_new",
        )
        for global_step in args.global_steps:
            run.log({"global_step": global_step, **payload})
        run.summary.update(metadata)
        run.finish()
        print(f"Logged {run_name} to {args.entity}/{args.project}.")


if __name__ == "__main__":
    main()
