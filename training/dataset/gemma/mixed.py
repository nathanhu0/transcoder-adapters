import random
from typing import Protocol

from torch.utils.data import Dataset
from training.dataset.types import DatasetItem


class SizedDataset(Protocol):
    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> DatasetItem: ...


class MixedDataset(Dataset):
    """Randomly interleaves multiple datasets according to specified weights.

    On each access, picks a source dataset with probability proportional to the
    given weights, then indexes into that dataset.  The total length equals the
    sum of all constituent dataset lengths.
    """

    def __init__(
        self,
        datasets: tuple[SizedDataset, ...],
        weights: tuple[float, ...],
        *,
        seed: int = 80,
    ):
        """Initialize the mixed dataset.

        Args:
            datasets: List of Dataset instances to mix.
            weights: Sampling weight for each dataset (need not sum to 1).
            seed: Random seed for reproducible mixing.
        """
        if len(datasets) != len(weights):
            raise ValueError(
                f"Got {len(datasets)} datasets but {len(weights)} weights"
            )

        self.datasets = datasets
        total_weight = sum(weights)
        self.weights = [w / total_weight for w in weights]

        # Pre-compute assignments for deterministic, shuffle-safe indexing.
        # Each global index maps to (dataset_index, local_index).
        rng = random.Random(seed)
        total = sum(len(d) for d in datasets)
        ds_indices = rng.choices(range(len(datasets)), weights=self.weights, k=total)

        # Track per-dataset cursors so each dataset is sampled uniformly.
        local_pools: list[list[int]] = []
        for ds in datasets:
            pool = list(range(len(ds)))
            rng.shuffle(pool)
            local_pools.append(pool)
        cursors = [0] * len(datasets)

        self._map: list[tuple[int, int]] = []
        for ds_idx in ds_indices:
            pool = local_pools[ds_idx]
            local_idx = pool[cursors[ds_idx] % len(pool)]
            cursors[ds_idx] += 1
            self._map.append((ds_idx, local_idx))

    def __len__(self) -> int:
        return len(self._map)

    def __getitem__(self, idx: int) -> DatasetItem:
        ds_idx, local_idx = self._map[idx]
        return self.datasets[ds_idx][local_idx]
