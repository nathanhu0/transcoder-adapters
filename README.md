# Transcoder Adapters for Reasoning-Model Diffing

Code for training, analyzing, and serving transcoder adapters, as described in [Transcoder Adapters for Reasoning-Model Diffing](https://transcoder-adapters.github.io/).

Transcoder adapters learn an interpretable approximation of the difference in MLP computation before and after fine-tuning. Each transformer layer's MLP is extended with a sparse parallel branch: `output = original_mlp(x) + dec(relu(enc(x)))`. Only the adapter parameters are trained; the base model is frozen.

## Pretrained Checkpoints

Five checkpoints trained on DeepSeek-R1-Distill-Qwen-7B with different L1 sparsity weights:

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

See `demo/` for full examples including adapter on/off toggling and vLLM serving.

## Training

Install dependencies, then run:

```bash
pip install -r requirements.txt
python -m training.train --config training/configs/r1_distil_7b.yaml
```

Training requires two models from the same architecture family: a **base model** (e.g., `Qwen/Qwen2.5-Math-7B`) and a **target model** (e.g., `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`). The adapter learns to approximate the MLP difference between them using bridging losses (KL divergence at sampled layer cutoffs + NMSE activation matching).

Configs are YAML files — see `training/configs/` for examples. Key settings:

- `model_name`: base model (MLP weight donor)
- `bridging.reference_model_path`: target model (distillation target)
- `transcoder.n_features`: width of the sparse adapter (e.g., 8192)
- `transcoder.l1_weight`: L1 sparsity penalty — higher values produce sparser adapters with fewer active features per token
- `learning_rate`: we used 8e-4 for all experiments

The two most important hyperparameters — `learning_rate` and `l1_weight` — can be overridden from the command line without editing the config:
```bash
python -m training.train --config training/configs/r1_distil_7b.yaml --learning_rate 1e-3 --l1_weight 0.01
```

Checkpoints are saved as standard HuggingFace format via `model.save_pretrained()`. Training logs to [Weights & Biases](https://wandb.ai) by default.

## Data

Training uses a stratified subset of [OpenThoughts3-1.2M](https://huggingface.co/datasets/open-thoughts/OpenThoughts3-1.2M) (70% math, 20% code, 10% science), filtered for complete reasoning traces under 10k tokens.

- **Pre-built splits**: [nathu0/transcoder-adapters-openthoughts3-stratified-55k](https://huggingface.co/datasets/nathu0/transcoder-adapters-openthoughts3-stratified-55k) (49,952 train / 4,996 val)
- **Reproduction script**: `misc_scripts/filter_openthoughts_stratified.py`

## Analysis

For feature activation collection, classification, auto-interp, and attribution graph generation, see [`analysis/README.md`](analysis/README.md).

## Repo Structure

- `models/` — Qwen2 + transcoder model class for training/inference (`qwen2_transcoder.py`) and vLLM serving (`qwen2_transcoder_vllm.py`)
- `training/` — Training script, config system, losses, dataset loader, and experiment configs
- `analysis/` — Feature analysis pipeline: activation collection, LLM-judge classification, auto-interp, and RelP attribution graphs
- `demo/` — Example scripts for generation with adapters on/off (transformers and vLLM)
- `misc_scripts/` — Dataset filtering, hybrid model creation

## Citing

If transcoder adapters or this repository is useful in your own research, you can use the following BibTeX entry:

```
@misc{hu2026transcoderadaptersreasoningmodeldiffing,
  title={Transcoder Adapters for Reasoning-Model Diffing},
  author={Nathan Hu and Jake Ward and Thomas Icard and Christopher Potts},
  year={2026},
  eprint={2602.20904},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2602.20904},
}
```
