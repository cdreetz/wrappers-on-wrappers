import torch
import time
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
    # Benchmark your model
    print("\n=== Your Custom Model ===")
    start_time = time.time()
    output = generate_text(qwen, tokenizer, prompt, max_length=100)
    end_time = time.time()
    
    # Calculate tokens generated
    input_tokens = len(tokenizer.encode(prompt))
    output_tokens = len(tokenizer.encode(output))
    generated_tokens = output_tokens - input_tokens
    
    custom_time = end_time - start_time
    custom_tok_per_sec = generated_tokens / custom_time if custom_time > 0 else 0
    
    print(f"Output: {output}")
    print(f"Time: {custom_time:.2f}s")
    print(f"Generated tokens: {generated_tokens}")
    print(f"Tokens/sec: {custom_tok_per_sec:.2f}")
    
    # Benchmark official model
    print("\n=== Official HuggingFace Model ===")
    inputs = tokenizer(prompt, return_tensors="pt")
    
    start_time = time.time()
    hf_output = hf_model.generate(**inputs, max_length=100, do_sample=False)
    end_time = time.time()
    
    hf_text = tokenizer.decode(hf_output[0], skip_special_tokens=True)
    hf_generated_tokens = len(hf_output[0]) - len(inputs.input_ids[0])
    
    hf_time = end_time - start_time
    hf_tok_per_sec = hf_generated_tokens / hf_time if hf_time > 0 else 0
    
    print(f"Output: {hf_text}")
    print(f"Time: {hf_time:.2f}s")
    print(f"Generated tokens: {hf_generated_tokens}")
    print(f"Tokens/sec: {hf_tok_per_sec:.2f}")
    
    # Comparison
    print(f"\n=== Comparison ===")
    print(f"Custom model: {custom_tok_per_sec:.2f} tok/s")
    print(f"Official model: {hf_tok_per_sec:.2f} tok/s")
    speedup = hf_tok_per_sec / custom_tok_per_sec if custom_tok_per_sec > 0 else float('inf')
    print(f"Official is {speedup:.2f}x faster" if speedup > 1 else f"Custom is {1/speedup:.2f}x faster")
