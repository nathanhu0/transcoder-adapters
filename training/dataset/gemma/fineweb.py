from torch.utils.data import Dataset

from datasets import load_dataset, Dataset as HFDataset
from training.dataset.types import DatasetItem


class FineWebDataset(Dataset):
    """Dataset for FineWeb pretraining data.

    Loads plain-text documents via the HuggingFace datasets library and applies
    language modeling loss on all tokens.

    Schema (science-of-finetuning/fineweb-1m-sample):
        text, id, dump, url, date, file_path, language, language_score, token_count
    """

    def __init__(
        self,
        data_path: str,
        *,
        tokenizer,
        max_length: int = 8192,
        truncate: bool = False,
        split: str = "train",
        text_field: str = "text",
    ):
        """Initialize the dataset.

        Args:
            data_path: HuggingFace dataset identifier
                       (e.g. "science-of-finetuning/fineweb-1m-sample").
            tokenizer: HuggingFace tokenizer.
            max_length: Maximum sequence length in tokens.
            truncate: If True, truncate long sequences. If False, raise on overflow.
            split: Dataset split to load (e.g. "train", "validation").
            text_field: Column containing the document text.
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.truncate = truncate
        self.text_field = text_field

        print(f"Loading FineWeb data: {data_path} (split={split})")
        self.ds: HFDataset = load_dataset(data_path, split=split) # pyright: ignore[reportAttributeAccessIssue]
        print(f"Loaded {len(self.ds)} examples")

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> DatasetItem:
        text = self.ds[idx][self.text_field]
        input_ids = self.tokenizer.encode(text, add_special_tokens=True)

        # Truncate if needed
        original_length = len(input_ids)
        truncated = False
        if len(input_ids) > self.max_length:
            if self.truncate:
                input_ids = input_ids[:self.max_length]
                truncated = True
            else:
                raise ValueError(
                    f"Sequence length {len(input_ids)} > max_length {self.max_length}"
                )

        # Pretraining: loss on all tokens
        labels = input_ids.copy()

        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
            "truncated": truncated,
            "original_length": original_length,
        }
