from pathlib import Path
from types import SimpleNamespace

import torch
import yaml

from src.wrappers.pretrain_wrappers import _build_dino_augmentation


def test_hybrid_null_local_views_use_configured_batched_sixty_percent_crops():
    config = yaml.safe_load(
        Path("configs/pretrain/dion_hybrid_distractor_null_local_100pct.yaml").read_text()
    )["dion"]
    augmentation = _build_dino_augmentation(config, SimpleNamespace(max_peaks=200))
    peaks = torch.arange(2 * 10 * 2, dtype=torch.float32).reshape(2, 10, 2)
    lengths = torch.tensor([10, 7])

    torch.manual_seed(7)
    crops = augmentation(peaks, lengths, rand_size=False)
    local_crops = crops[config["num_global_crops"]:]

    assert config["selection_mode"] == "random_batched"
    assert config["local_crops_scale"] == [0.6, 0.6]
    assert len(local_crops) >= 2
    for _, padding in local_crops[:2]:
        assert (~padding).sum(dim=1).tolist() == [6, 4]
