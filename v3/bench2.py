import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from configuration_qwen2 import Qwen2Config
from model import Qwen2ForCausalLM, generate_text

def benchmark_my_qwen(prompt, hf_model, tokenizer, max_tokens=100):
    hf_model.cpu()
    config = Qwen2Config()
    qwen = Qwen2ForCausalLM(config).cuda()
    qwen.load_state_dict(hf_model.state_dict())
    qwen.eval()
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Warmup
    _ = generate_text(qwen, tokenizer, text, max_length=10)
    torch.cuda.synchronize()
    
    # Benchmark
    start_time = time.time()
    output = generate_text(qwen, tokenizer, text, max_length=max_tokens)
    torch.cuda.synchronize()
    end_time = time.time()
    
    # Count generated tokens
    input_tokens = len(tokenizer.encode(text))
    total_tokens = len(tokenizer.encode(output))
    generated_tokens = total_tokens - input_tokens
    
    elapsed_time = end_time - start_time
    tokens_per_second = generated_tokens / elapsed_time if elapsed_time > 0 else 0
    
    del qwen
    torch.cuda.empty_cache()
    
    return output, generated_tokens, elapsed_time, tokens_per_second

def benchmark_hf_qwen(prompt, hf_model, tokenizer, max_tokens=100):
    hf_model.cuda()
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(hf_model.device)
    
    # Warmup
    _ = hf_model.generate(**model_inputs, max_new_tokens=10, do_sample=False)
    torch.cuda.synchronize()
    
    # Benchmark
    start_time = time.time()
    generated_ids = hf_model.generate(
        **model_inputs,
        max_new_tokens=max_tokens,
        do_sample=False
    )
    torch.cuda.synchronize()
    end_time = time.time()
    
    # Extract generated tokens
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    generated_tokens = len(generated_ids[0])
    elapsed_time = end_time - start_time
    tokens_per_second = generated_tokens / elapsed_time if elapsed_time > 0 else 0
    
    return response, generated_tokens, elapsed_time, tokens_per_second

def run_benchmark(prompt, max_tokens=100, num_runs=3):
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    hf_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print(f"Benchmarking with prompt: '{prompt}'")
    print(f"Max tokens: {max_tokens}, Runs: {num_runs}")
    print("=" * 80)
    
    # Benchmark custom implementation
    my_times = []
    my_tok_per_sec = []
    print("Benchmarking custom implementation...")
    
    for i in range(num_runs):
        output, gen_tokens, elapsed, tok_per_sec = benchmark_my_qwen(prompt, hf_model, tokenizer, max_tokens)
        my_times.append(elapsed)
        my_tok_per_sec.append(tok_per_sec)
        print(f"  Run {i+1}: {gen_tokens} tokens in {elapsed:.3f}s = {tok_per_sec:.2f} tok/s")
        if i == 0:  # Show output from first run
            print(f"  Output: {output[:100]}..." if len(output) > 100 else f"  Output: {output}")
    
    print()
    
    # Benchmark HuggingFace implementation
    hf_times = []
    hf_tok_per_sec = []
    print("Benchmarking HuggingFace implementation...")
    
    for i in range(num_runs):
        output, gen_tokens, elapsed, tok_per_sec = benchmark_hf_qwen(prompt, hf_model, tokenizer, max_tokens)
        hf_times.append(elapsed)
        hf_tok_per_sec.append(tok_per_sec)
        print(f"  Run {i+1}: {gen_tokens} tokens in {elapsed:.3f}s = {tok_per_sec:.2f} tok/s")
        if i == 0:  # Show output from first run
            print(f"  Output: {output[:100]}..." if len(output) > 100 else f"  Output: {output}")
    
    print()
    print("=" * 80)
    print("RESULTS:")
    print(f"Custom Implementation:")
    print(f"  Average: {sum(my_tok_per_sec)/len(my_tok_per_sec):.2f} tok/s")
    print(f"  Best: {max(my_tok_per_sec):.2f} tok/s")
    print(f"  Time: {sum(my_times)/len(my_times):.3f}s avg")
    
    print(f"HuggingFace Implementation:")
    print(f"  Average: {sum(hf_tok_per_sec)/len(hf_tok_per_sec):.2f} tok/s")
    print(f"  Best: {max(hf_tok_per_sec):.2f} tok/s")
    print(f"  Time: {sum(hf_times)/len(hf_times):.3f}s avg")
    
    speedup = (sum(my_tok_per_sec)/len(my_tok_per_sec)) / (sum(hf_tok_per_sec)/len(hf_tok_per_sec))
    print(f"Speedup: {speedup:.2f}x ({'faster' if speedup > 1 else 'slower'})")

if __name__ == "__main__":
    # Quick benchmark
    run_benchmark("Explain quantum computing in simple terms.", max_tokens=50, num_runs=3)
    
    # Longer benchmark
    print("\n" + "=" * 80)
    print("LONGER GENERATION BENCHMARK")
    run_benchmark("Write a detailed explanation of machine learning.", max_tokens=200, num_runs=2)
