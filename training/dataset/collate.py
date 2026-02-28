import torch

def collate_fn(examples, tokenizer):
    """Simple collate function for batching examples."""
    # Extract sequences
    input_ids = [ex["input_ids"] for ex in examples]
    labels = [ex["labels"] for ex in examples]

    # Pad input_ids and attention_mask
    batch = tokenizer.pad(
        {"input_ids": input_ids},
        padding=True,
        return_tensors="pt"
    )

    # Manually pad labels with -100
    max_length = batch["input_ids"].shape[1]
    padded_labels = []

    for label_seq in labels:
        padded = label_seq + [-100] * (max_length - len(label_seq))
        padded_labels.append(padded)

    batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)

    # Pass through truncation stats
    batch["truncated"] = [ex["truncated"] for ex in examples]
    batch["original_length"] = [ex["original_length"] for ex in examples]

    return batch
