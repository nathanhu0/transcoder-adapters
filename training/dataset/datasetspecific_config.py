from dataclasses import dataclass
from enum import Enum
from typing import ClassVar


class DatasetType(Enum):
    OPEN_THOUGHTS = "open_thoughts"
    FINEWEB_LMYSYSCHAT_MIXED = "fineweb_lmysyschat_mixed"


@dataclass(frozen=True)
class DatasetSpecificConfig:
    dataset_type: ClassVar[DatasetType]
