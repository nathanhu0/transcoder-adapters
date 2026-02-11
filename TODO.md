# TODO

## Verified
- [x] Training runs end-to-end (`python -m training.train --config training/configs/r1_distil_7b_debug.yaml`)
- [x] Loss magnitudes match old codebase (required re-init of transcoder weights after `from_pretrained`)

## To Verify
- [ ] vLLM weight loading (`models/qwen2_transcoder_vllm.py`) — test with a trained checkpoint
- [ ] Checkpoint saving/loading round-trip (save with `save_pretrained`, reload with `from_pretrained`)
- [ ] Full training run reproduces old results (compare wandb curves)

## To Implement
- [ ] Migrate feature activation collection code to `analysis/`
- [ ] Auto-interp pipeline
- [ ] Feature dashboards
- [ ] Host attribution graphs and dashboards
- [ ] Host model checkpoints and datasets on HuggingFace
