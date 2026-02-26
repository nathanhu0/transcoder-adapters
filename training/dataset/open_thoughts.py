from torch.utils.data import Dataset
from typing import Any, Literal
import json


# DeepSeek R1 Distill format tokens
DEEPSEEK_USER_TOKEN = "<｜User｜>"
DEEPSEEK_ASSISTANT_TOKEN = "<｜Assistant｜>"

# Qwen/QwQ chat format tokens
QWEN_IM_START = "<|im_start|>"
QWEN_IM_END = "<|im_end|>"


class OpenThoughtsDataset(Dataset):
    """Dataset for loading OpenThoughts reasoning traces.

    Supports multiple formats for both SFT and bridging training.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 8192,
        # Format
        format: Literal["tokenizer", "deepseek"] = "tokenizer",
        # Truncation
        truncate: bool = False,
        # Labels
        loss_on_prompt: bool = False,
        # Legacy params
        split: str = "train",
        filter_length: bool = False,
    ):
        """Initialize the dataset.

        Args:
            data_path: Path to local JSONL file
            tokenizer: Tokenizer for processing text
            max_length: Maximum sequence length

            Formatting:
                format: "tokenizer" uses apply_chat_template (default, backwards compatible)
                        "deepseek" uses explicit DeepSeek R1 format (preserves <think>)

            Truncation:
                truncate: If True, truncate to max_length
                          If False, assert length <= max_length (default)

            Labels:
                loss_on_prompt: If True, include prompt tokens in loss (for bridging).
                                Padding is still masked.

            Legacy:
                split: Dataset split (unused, kept for compatibility)
                filter_length: Filter by length at load time
        """
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.format = format
        self.truncate = truncate
        self.loss_on_prompt = loss_on_prompt
        self.split = split
        self.filter_length = filter_length
        if self.truncate and self.filter_length:
            print('truncating enabled, disabling filter_length')
            self.filter_length = False
        self.examples = []
        self._load_data()

    def _load_data(self):
        """Load filtered JSONL data from local path or HuggingFace (hf://repo/path)."""
        print(f"Loading data from {self.data_path}")

        all_examples = []
        if self.data_path.startswith("hf://"):
            from huggingface_hub import hf_hub_download
            # Parse hf://org/repo/path/to/file.jsonl
            hf_path = self.data_path[len("hf://"):]
            parts = hf_path.split("/", 2)  # org, repo, filepath
            repo_id = f"{parts[0]}/{parts[1]}"
            filepath = parts[2]
            local_path = hf_hub_download(repo_id=repo_id, filename=filepath, repo_type="dataset")
        else:
            local_path = self.data_path

        with open(local_path, 'r') as f:
            for line in f:
                example = json.loads(line.strip())
                all_examples.append(example)

        if self.filter_length:
            print(f"Filtering {len(all_examples)} examples by length...")
            for example in all_examples:
                formatted_text = self.format_example(example)
                tokens = self.tokenizer(formatted_text, add_special_tokens=True)
                if len(tokens['input_ids']) <= self.max_length:
                    self.examples.append(example)
            print(f"Kept {len(self.examples)}/{len(all_examples)} after filtering")
        else:
            self.examples = all_examples
            print(f"Loaded {len(self.examples)} examples (no length filtering)")

    def __len__(self) -> int:
        """Return the number of examples."""
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a single example."""
        example = self.examples[idx]
        conversations = example['conversations']
        prompt = conversations[0]['value']
        response = conversations[1]['value']

        # Format and tokenize
        if self.format == "deepseek":
            # Normalize "<think> " to "<think>\n" to match DeepSeek's expected format
            if response.startswith("<think> "):
                response = "<think>\n" + response[len("<think> "):]
            text = f"{DEEPSEEK_USER_TOKEN}{prompt}{DEEPSEEK_ASSISTANT_TOKEN}{response}"
            input_ids = self.tokenizer.encode(text, add_special_tokens=True)
            # Prompt boundary: tokenize just the prompt portion
            prompt_text = f"{DEEPSEEK_USER_TOKEN}{prompt}{DEEPSEEK_ASSISTANT_TOKEN}"
            prompt_len = len(self.tokenizer.encode(prompt_text, add_special_tokens=True))
        elif self.format == "qwen":
            # Explicit Qwen/QwQ format: <|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n...<|im_end|>
            if response.startswith("<think> "):
                response = "<think>\n" + response[len("<think> "):]
            text = f"{QWEN_IM_START}user\n{prompt}{QWEN_IM_END}\n{QWEN_IM_START}assistant\n{response}{QWEN_IM_END}\n"
            input_ids = self.tokenizer.encode(text, add_special_tokens=True)
            # Prompt boundary
            prompt_text = f"{QWEN_IM_START}user\n{prompt}{QWEN_IM_END}\n{QWEN_IM_START}assistant\n"
            prompt_len = len(self.tokenizer.encode(prompt_text, add_special_tokens=True))
        else:
            text = self.format_example(example)
            input_ids = self.tokenizer.encode(text, add_special_tokens=True)
            # Prompt boundary: use chat template
            prompt_len = len(self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True
            ))

        # Truncate if needed
        original_length = len(input_ids)
        truncated = False
        if len(input_ids) > self.max_length:
            if self.truncate:
                input_ids = input_ids[:self.max_length]
                truncated = True
            else:
                raise ValueError(f"Sequence {len(input_ids)} > max_length {self.max_length}")

        # Labels: include prompt or mask it
        if self.loss_on_prompt:
            labels = input_ids.copy()
        else:
            labels = [-100] * min(prompt_len, len(input_ids)) + input_ids[prompt_len:]

        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
            "truncated": truncated,
            "original_length": original_length,
        }

    def format_example(self, example: dict[str, Any]) -> str:
        """Format a single example into the training format.

        Args:
            example: Raw example from dataset

        Returns:
            Formatted string ready for tokenization
        """
        conversations = example['conversations']
        user_message = conversations[0]['value']
        assistant_message = conversations[1]['value']

        # Use chat template format
        messages = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message}
        ]

        formatted = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        return formatted
