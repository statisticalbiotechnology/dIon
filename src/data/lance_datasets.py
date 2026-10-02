from __future__ import annotations

from typing import Any, List

from torch.utils.data import Dataset


class SafeLanceDataset(Dataset):
    """Map-style dataset backed by lance.torch.data.SafeLanceDataset."""

    def __init__(
        self, uri: str, columns: list[str] | None = None, include_index: bool = False
    ) -> None:
        from lance.torch.data import SafeLanceDataset as _SafeLanceDataset

        self.uri = uri
        if columns is not None:
            required = ["mz_array", "intensity_array"]
            columns = list(dict.fromkeys([*required, *columns]))
        self.columns = columns
        self.include_index = bool(include_index)
        self._base = _SafeLanceDataset(uri)

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self._base[index]
        if self.columns is not None:
            cols = set(self.columns)
            item = {k: v for k, v in item.items() if k in cols}
        if getattr(self, "include_index", False):
            item["index"] = int(index)
        return item

    def __getitems__(self, indices: List[int]) -> list[dict[str, Any]]:
        items = self._base.__getitems__(indices)
        if self.columns is not None:
            cols = set(self.columns)
            items = [{k: v for k, v in item.items() if k in cols} for item in items]
        if getattr(self, "include_index", False):
            for item, index in zip(items, indices, strict=True):
                item["index"] = int(index)
        return items
