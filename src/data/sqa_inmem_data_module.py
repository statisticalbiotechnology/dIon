import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule


class SpectrumQualityDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.df = self.df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return {
            "index": torch.as_tensor(idx, dtype=torch.long),
            "scan_num": row["scan_num"],
            "precursor_mz": torch.as_tensor(row["precursor_mz"], dtype=torch.float32),
            "precursor_charge": torch.as_tensor(
                row["precursor_charge"], dtype=torch.float32
            ),
            "mz_array": torch.as_tensor(row["mz_array"], dtype=torch.float32),
            "intensity_array": torch.as_tensor(
                row["intensity_array"], dtype=torch.float32
            ),
            "quality": torch.as_tensor(row["label"], dtype=torch.long),
        }


class SpectrumQualityDataModule(LightningDataModule):
    def __init__(self, data_root, batch_size, collate_fn, seed=0, num_workers=0):
        super().__init__()
        self.data_root = data_root
        self.batch_size = batch_size
        self.collate_fn = collate_fn
        self.seed = seed
        self.epoch = 0
        self.num_workers = num_workers
        self.datasets = {}

    def setup(self, stage=None):
        for split in ["train", "val", "test"]:
            path = os.path.join(self.data_root, f"{split}.parquet")
            df = pd.read_parquet(path)
            self.datasets[split] = SpectrumQualityDataset(df)

    def _dl(self, split, shuffle):
        g = torch.Generator()
        # The generic Lance callback advances this epoch hook. Keep training
        # shuffles reproducible while avoiding the identical order each epoch.
        g.manual_seed(self.seed + self.epoch)
        return DataLoader(
            self.datasets[split],
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=self.collate_fn,
            generator=g,
            num_workers=self.num_workers,
            pin_memory=True,
            # Evaluation must cover every labelled spectrum. Dropping the last
            # batch is only useful for shuffled training batches.
            drop_last=shuffle,
        )

    def set_epoch(self, epoch: int) -> None:
        """Match the datamodule contract used by LanceSamplerEpochCallback."""
        self.epoch = int(epoch)

    def train_dataloader(self):
        return self._dl("train", shuffle=True)

    def val_dataloader(self):
        return self._dl("val", shuffle=False)

    def test_dataloader(self):
        return self._dl("test", shuffle=False)
