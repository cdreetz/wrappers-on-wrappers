from transformers import AutoModel, AutoTokenizer
import torch
import inspect

def inspect_model():
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    model = AutoModel.from_pretrained(model_name)
    print("Model:\n")
    print(model)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    embd_layer = model.embed_tokens
    print("embedding layer:")
    print(inspect.getsource(type(embd_layer)))

    rmsnorm = model.layers[0].input_layernorm
    print("rmsnorm input layer:")
    print(inspect.getsource(type(rmsnorm)))


    attention = model.layers[0].self_attn
    print("self attn layer:")
    print(inspect.getsource(type(attention)))

    mlp = model.layers[0].mlp
    print("mlp/ffn layer:")
    print(inspect.getsource(type(mlp)))

    decoder_layer = model.layers[0]
    print("decoder layer:")
    print(inspect.getsource(type(decoder_layer)))

    final_norm = model.norm
    print("final norm layer:")
    print(inspect.getsource(type(final_norm)))


    #torch.save(model.state_dict(), 'qwen_weights.pth')


inspect_model()
