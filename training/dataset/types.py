from typing import TypedDict


class DatasetItem(TypedDict):
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]
    truncated: bool
    original_length: int
