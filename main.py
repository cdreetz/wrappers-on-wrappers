import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from configuration_qwen2 import Qwen2Config
from model import Qwen2ForCausalLM, generate_text

config = Qwen2Config()
hf_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

qwen = Qwen2ForCausalLM(config)
missing, unexpected = qwen.load_state_dict(hf_model.state_dict(), strict=False)
qwen.eval()

prompt = "How many r's are there in strawberry?"
print(f"Starting generation for: '{prompt}'")

with torch.no_grad():
    output = generate_text(qwen, tokenizer, prompt, max_length=100)
    print(f"Output: {output}")

    inputs = tokenizer(prompt, return_tensors="pt")
    hf_output = hf_model.generate(**inputs, max_length=100, do_sample=False)
    hf_text = tokenizer.decode(hf_output[0], skip_special_tokens=True)
    print("Official model output: {hf_text}")
