"""Load a transcoder adapter checkpoint from HuggingFace with vLLM."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.qwen2_transcoder_vllm import register_vllm_transcoder
register_vllm_transcoder()

from vllm import LLM, SamplingParams

#%% Load model
MODEL = "nathu0/transcoder-adapters-R1-Distill-Qwen-7B-l1w0.001-l0-1.4"

llm = LLM(model=MODEL, trust_remote_code=True, dtype="bfloat16")

#%% Generate
question = "What is the 10th digit of pi?"
prompt = f"<｜User｜>{question}<｜Assistant｜><think>\n"

print(f"----- Prompt: {question} -----\n")

output = llm.generate([prompt], SamplingParams(max_tokens=3000, temperature=0.7))

print("----- Response -----")
print(output[0].outputs[0].text)
print("--------------------")
print("\nFinished. vLLM may print engine shutdown errors below -- these are harmless.")
