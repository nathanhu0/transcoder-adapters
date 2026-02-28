from dataclasses import dataclass
from typing import ClassVar

from training.dataset.datasetspecific_config import DatasetSpecificConfig, DatasetType
from .types import DataFormat


@dataclass
class OpenThoughtsConfig(DatasetSpecificConfig):
    dataset_type: ClassVar[DatasetType] = DatasetType.OPEN_THOUGHTS

    data_path: str
    data_format: DataFormat

    max_seq_length: int = 10000

    val_data_path: str | None = None
