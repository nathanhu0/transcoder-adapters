import json


def resolve_data_path(data_path: str) -> str:
    """Resolve a data path, downloading from HuggingFace Hub if needed.

    Supports:
        - Local paths: "/path/to/file.jsonl"
        - HuggingFace Hub: "hf://org/repo/path/to/file.jsonl"

    Returns:
        Local filesystem path to the file.
    """
    if data_path.startswith("hf://"):
        from huggingface_hub import hf_hub_download
        hf_path = data_path[len("hf://"):]
        parts = hf_path.split("/", 2)  # org, repo, filepath
        repo_id = f"{parts[0]}/{parts[1]}"
        filepath = parts[2]
        return hf_hub_download(repo_id=repo_id, filename=filepath, repo_type="dataset")
    return data_path


def load_jsonl(data_path: str) -> list[dict]:
    """Load examples from a JSONL file (local or hf:// path)."""
    local_path = resolve_data_path(data_path)
    examples = []
    with open(local_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples
