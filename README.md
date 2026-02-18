# Transcoder Adapters

Sparse transcoder adapters for bridging distillation. Trains a parallel ReLU-based sparse branch on each MLP layer to bridge the gap between a base model (Qwen2.5-Math-7B) and a reference model (DeepSeek-R1-Distill-Qwen-7B), using layer-wise bridging losses to encourage compatibility.

## Repo Structure

```
transcoder_adapters/
├── models/
│   ├── qwen2_transcoder.py          # Qwen2 + transcoder model (training & inference)
│   └── qwen2_transcoder_vllm.py     # vLLM-compatible implementation
├── training/
│   ├── train.py                     # Main training script
│   ├── config.py                    # Experiment config dataclasses + YAML loading
│   ├── configs/
│   │   └── r1_distil_7b.yaml        # Default experiment config
│   ├── losses.py                    # KL, LM, and NMSE loss functions
│   ├── forward_utils.py             # Mixed forward passes for bridging
│   └── dataset.py                   # OpenThoughts dataset loader
├── demo/
│   ├── adapter_generation_transformers.py  # Load from HF + generate (adapters on/off)
│   └── adapter_generation_vllm.py          # Load from HF + generate with vLLM
├── misc_scripts/
│   └── filter_openthoughts_stratified.py   # Dataset reproduction script
├── analysis/                        # TODO: activation collection, auto-interp, dashboards
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Training

```bash
python -m training.train --config training/configs/r1_distil_7b.yaml
```

Override hyperparameters from the command line:
```bash
python -m training.train --config training/configs/r1_distil_7b.yaml --learning_rate 1e-3 --l1_weight 0.01
```

## How It Works

Each transformer layer's MLP is extended with a transcoder branch:
```
output = original_mlp(x) + dec(relu(enc(x)))
```

- `enc`: d_model -> n_features (with bias, ReLU activation)
- `dec`: n_features -> d_model (initialized to zero for zero initial contribution)

Only transcoder parameters are trained; the base model is frozen. Training uses bridging losses (KL divergence at sampled layer cutoffs) plus optional NMSE activation matching to encourage layer-wise compatibility with the reference model.

Checkpoints are saved as standard HuggingFace format via `model.save_pretrained()` -- no conversion step needed.

## Pretrained Checkpoints

Five checkpoints trained with different L1 sparsity weights (and resulting L0 activation counts):

| L1 Weight | Val L0 | HuggingFace |
|-----------|--------|-------------|
| 0.01      | 10.3   | [nathu0/transcoder-adapters-R1-Distill-Qwen-7B-l1w0.01-l0-10.3](https://huggingface.co/nathu0/transcoder-adapters-R1-Distill-Qwen-7B-l1w0.01-l0-10.3) |
| 0.003     | 4.3    | [nathu0/transcoder-adapters-R1-Distill-Qwen-7B-l1w0.003-l0-4.3](https://huggingface.co/nathu0/transcoder-adapters-R1-Distill-Qwen-7B-l1w0.003-l0-4.3) |
| 0.001     | 1.4    | [nathu0/transcoder-adapters-R1-Distill-Qwen-7B-l1w0.001-l0-1.4](https://huggingface.co/nathu0/transcoder-adapters-R1-Distill-Qwen-7B-l1w0.001-l0-1.4) |
| 0.0003    | 0.4    | [nathu0/transcoder-adapters-R1-Distill-Qwen-7B-l1w0.0003-l0-0.4](https://huggingface.co/nathu0/transcoder-adapters-R1-Distill-Qwen-7B-l1w0.0003-l0-0.4) |
| 0.0001    | 0.1    | [nathu0/transcoder-adapters-R1-Distill-Qwen-7B-l1w0.0001-l0-0.1](https://huggingface.co/nathu0/transcoder-adapters-R1-Distill-Qwen-7B-l1w0.0001-l0-0.1) |

Load with the custom model class (not compatible with AutoModel):

```python
from models.qwen2_transcoder import Qwen2ForCausalLMWithTranscoder

model = Qwen2ForCausalLMWithTranscoder.from_pretrained(
    "nathu0/transcoder-adapters-R1-Distill-Qwen-7B-l1w0.001-l0-1.4",
    torch_dtype=torch.bfloat16, device_map="auto",
)
```

See `demo/adapter_generation_transformers.py` and `demo/adapter_generation_vllm.py` for full examples including adapter on/off toggling.

## vLLM

We use a fork of [evalchemy](https://github.com/mlfoundations/evalchemy) for evaluations. To aid in future work, we provide a vLLM implementation of transcoder adapters in `models/qwen2_transcoder_vllm.py`. See `demo/adapter_generation_vllm.py` for usage.

## Data

Training uses a stratified subset of [OpenThoughts3-1.2M](https://huggingface.co/datasets/open-thoughts/OpenThoughts3-1.2M) (70% math, 20% code, 10% science), filtered for complete reasoning traces under 10k tokens.

- **Pre-built splits**: [nathu0/transcoder-adapters-openthoughts3-stratified-55k](https://huggingface.co/datasets/nathu0/transcoder-adapters-openthoughts3-stratified-55k) (49,952 train / 4,996 val)
- **Reproduction script**: `misc_scripts/filter_openthoughts_stratified.py`

## Related Repos

- **evalchemy** (fork): Evaluation framework with vLLM transcoder support
- **circuit-tracer** (fork): Circuit analysis for transcoder features
