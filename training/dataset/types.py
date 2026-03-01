from typing import Protocol, TypeVar, TypedDict


class DatasetItem(TypedDict):
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]
    truncated: bool
    original_length: int


T_co = TypeVar("T_co", covariant=True)


class SizedDataset(Protocol[T_co]):
    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> T_co: ...
