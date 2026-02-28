from enum import Enum
from typing import ClassVar, Protocol


class DatasetType(Enum):
    OPEN_THOUGHTS = "open_thoughts"
    FINEWEB_LMYSYSCHAT_MIXED = "fineweb_lmysyschat_mixed"


class DatasetSpecificConfig(Protocol):
    dataset_type: ClassVar[DatasetType]
