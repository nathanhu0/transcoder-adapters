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

## Inference with vLLM

The vLLM implementation (`models/qwen2_transcoder_vllm.py`) loads trained checkpoints directly. See the evalchemy repo for the full evaluation pipeline.

## Related Repos

- **evalchemy** (fork): Evaluation framework with vLLM transcoder support
- **circuit-tracer** (fork): Circuit analysis for transcoder features
