from src.probe_conditioning import get_probe_conditioning_modes


def test_each_probe_task_owns_its_conditioning_modes():
    config = {
        "embedding_evaluation": {
            "online_precursor_conditioning_modes": ["conditioned", "null"]
        },
        "dense_denovo_probe": {
            "online_precursor_conditioning_modes": ["conditioned"]
        },
    }

    assert get_probe_conditioning_modes(
        config, "embedding_evaluation", "conditioned"
    ) == ["conditioned", "null"]
    assert get_probe_conditioning_modes(
        config, "dense_denovo_probe", "null"
    ) == ["conditioned"]


def test_default_mode_applies_when_no_probe_mode_is_declared():
    assert get_probe_conditioning_modes(
        {"dense_denovo_probe": {}}, "dense_denovo_probe", "null"
    ) == ["null"]
