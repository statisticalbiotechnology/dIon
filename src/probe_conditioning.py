"""Task-scoped precursor-conditioning configuration for online probes."""

from __future__ import annotations


VALID_PRECURSOR_CONDITIONING_MODES = frozenset({"conditioned", "null"})


def get_probe_conditioning_modes(
    probing_config: dict, task_name: str, default_mode: str
) -> list[str]:
    """Resolve the precursor-conditioning modes declared by one probe task.

    ``online_precursor_conditioning_modes`` belongs to the individual task
    section. When omitted, the enclosing run's ``default_mode`` is used.
    """
    task_config = probing_config[task_name]
    if not isinstance(task_config, dict):
        raise ValueError(f"Probe configuration {task_name!r} must be a mapping.")
    modes = task_config.get("online_precursor_conditioning_modes", [default_mode])
    if not isinstance(modes, list) or not modes:
        raise ValueError(
            f"{task_name}.online_precursor_conditioning_modes must be a non-empty list."
        )
    if len(set(modes)) != len(modes):
        raise ValueError(
            f"{task_name}.online_precursor_conditioning_modes must not contain duplicates."
        )
    invalid_modes = set(modes) - VALID_PRECURSOR_CONDITIONING_MODES
    if invalid_modes:
        raise ValueError(
            f"{task_name}.online_precursor_conditioning_modes has invalid values "
            f"{sorted(invalid_modes)}; expected "
            f"{sorted(VALID_PRECURSOR_CONDITIONING_MODES)}."
        )
    return modes
