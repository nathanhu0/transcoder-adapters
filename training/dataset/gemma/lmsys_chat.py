from torch.utils.data import Dataset

from datasets import Dataset as HFDataset, load_dataset
from training.dataset.types import DatasetItem


class LMSYSChatDataset(Dataset):
    """Dataset for LMSYS-Chat multi-turn conversations.

    Loads conversations via the HuggingFace datasets library and optionally
    masks non-assistant tokens from the loss.

    Schema (lmsys/lmsys-chat-1m):
        conversation_id, model, conversation, turn, language, openai_moderation, redacted
    The ``conversation`` column is a list of {"role": str, "content": str} dicts.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 8192,
        truncate: bool = False,
        loss_on_prompt: bool = False,
        split: str = "train",
        conversation_field: str = "conversation",
    ):
        """Initialize the dataset.

        Args:
            data_path: HuggingFace dataset identifier (e.g. "lmsys/lmsys-chat-1m").
            tokenizer: HuggingFace tokenizer.
            max_length: Maximum sequence length in tokens.
            truncate: If True, truncate long sequences. If False, raise on overflow.
            loss_on_prompt: If True, compute loss on all tokens.
                            If False, mask user/system turns (loss on assistant only).
            split: Dataset split to load.
            conversation_field: Column containing the conversation list.
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.truncate = truncate
        self.loss_on_prompt = loss_on_prompt
        self.conversation_field = conversation_field

        print(f"Loading LMSYS-Chat data: {data_path} (split={split})")
        self.ds: HFDataset = load_dataset(data_path, split=split) # pyright: ignore[reportAttributeAccessIssue]. Since we're choosing a split, we know a dict isn't returned, and since streaming is not True, we know it's not an IterableDataset.
        print(f"Loaded {len(self.ds)} examples")

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> DatasetItem:
        conversation = self.ds[idx][self.conversation_field]

        # Tokenize full conversation via chat template
        input_ids = self.tokenizer.apply_chat_template(
            conversation, tokenize=True, add_generation_prompt=False
        )

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

        # Build labels
        if self.loss_on_prompt:
            labels = list(input_ids)
        else:
            labels = self._mask_non_assistant(conversation, input_ids)

        return {
            "input_ids": list(input_ids),
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
            "truncated": truncated,
            "original_length": original_length,
        }

    def _mask_non_assistant(
        self, conversation: list[dict], full_ids: list[int]
    ) -> list[int]:
        """Mask all non-assistant tokens with -100.

        For each assistant turn, finds its token span by comparing chat-template
        tokenizations of the conversation prefix with and without that turn.
        """
        labels = [-100] * len(full_ids)

        for i, msg in enumerate(conversation):
            if msg["role"] not in ("assistant", "model"):
                continue

            # Tokens up to (not including) this assistant turn, with generation
            # prompt so we get the assistant turn-start marker included.
            prefix = conversation[:i]
            prefix_ids = self.tokenizer.apply_chat_template(
                prefix, tokenize=True, add_generation_prompt=True
            )
            start = len(prefix_ids)

            # Tokens through the end of this assistant turn.
            through = conversation[: i + 1]
            through_ids = self.tokenizer.apply_chat_template(
                through, tokenize=True, add_generation_prompt=False
            )
            end = min(len(through_ids), len(full_ids))

            # Unmask the assistant span
            labels[start:end] = list(full_ids[start:end])

        return labels
