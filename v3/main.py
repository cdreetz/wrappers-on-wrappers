import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from configuration_qwen2 import Qwen2Config
from model import Qwen2ForCausalLM, generate_text


def try_my_qwen(prompt, hf_model, tokenizer):
    hf_model.cpu()

    config = Qwen2Config()
    qwen = Qwen2ForCausalLM(config).cuda()
    qwen.load_state_dict(hf_model.state_dict())
    qwen.to(hf_model.device)
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

    output = generate_text(qwen, tokenizer, text, max_length=100)

    del qwen
    torch.cuda.empty_cache()

    return output


def try_hf_qwen(prompt, hf_model, tokenizer):
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

    generated_ids = hf_model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response


if __name__ == "__main__":
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    prompt = "How many r's are there in strawberry?"
    hf_model = AutoModelForCausalLM.from_pretrained(model_name,torch_dtype="auto",device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"Starting generation for: '{prompt}'")
    print("=="*20)
    print("trying my qwen")
    my_qwen = try_my_qwen(prompt, hf_model, tokenizer)
    print(my_qwen)

    print("=="*20)
    print("trying hf qwen")
    hf_qwen = try_hf_qwen(prompt, hf_model, tokenizer)
    print(hf_qwen)
