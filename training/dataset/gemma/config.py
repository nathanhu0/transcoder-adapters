from dataclasses import dataclass
from typing import ClassVar, Literal

from training.dataset.datasetspecific_config import DatasetSpecificConfig, DatasetType


@dataclass(frozen=True)
class FineWebLMSysMixedConfig(DatasetSpecificConfig):
    dataset_type: ClassVar[DatasetType] = DatasetType.FINEWEB_LMYSYSCHAT_MIXED

    pretraining_datapath: str = "science-of-finetuning/fineweb-1m-sample"
    chat_conversations_datapath: str = "lmsys/lmsys-chat-1m"

    pretraining_max_seq_length: int = 8192
    chat_max_seq_length: int | Literal["pretraining_max_seq_length"] = "pretraining_max_seq_length"
