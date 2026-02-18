"""Load a transcoder adapter checkpoint from HuggingFace with transformers."""
#%%
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoTokenizer
from models.qwen2_transcoder import Qwen2ForCausalLMWithTranscoder

#%% Load model and tokenizer
MODEL = "nathu0/transcoder-adapters-R1-Distill-Qwen-7B-l1w0.001-l0-1.4"

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
model = Qwen2ForCausalLMWithTranscoder.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="auto")
model.eval()

#%% Generate with adapters ON (reasoning behavior)
question = "What is the 10th digit of pi?"
prompt = f"<｜User｜>{question}<｜Assistant｜><think>\n"
inputs = tokenizer(prompt, add_special_tokens=True, return_tensors="pt").to(model.device)

print(f"----- Prompt: {question} -----\n")

output = model.generate(**inputs, max_new_tokens=500, do_sample=True, temperature=0.7)

print("----- Response with adapters ON -----")
print(tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=False))
print("--------------------------------------\n\n")

#%% Generate with adapters OFF (hybrid: base mlp, reasoning elsewhere)
for layer in model.model.layers:
    layer.mlp.disable_transcoder = True

output = model.generate(**inputs, max_new_tokens=500, do_sample=True, temperature=0.7)

print("----- Response with adapters OFF -----")
print(tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=False))
print("--------------------------------------")
