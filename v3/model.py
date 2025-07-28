import torch
import torch.nn as nn
import triton
import triton.language as tl
from typing import Optional, Callable

from configuration_qwen2 import Qwen2Config

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def repeat_kv(hidden_states: torch.Tensor, n_rep:int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2,3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    
    return attn_output, attn_weights

def create_causal_mask(seq_len, device, dtype):
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=dtype), diagonal=1)
    mask = torch.where(mask == 1, float('-inf'), 0.0)
    return mask.unsqueeze(0).unsqueeze(0)

@triton.jit
def rmsnorm_kernel(
    x_ptr, weight_ptr, output_ptr,
    n_rows, n_cols, eps,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    x_row = tl.load(x_ptr + row_idx * n_cols + col_offsets, mask=mask)

    x_squared = x_row * x_row
    var = tl.sum(x_squared, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(var + eps)

    weight = tl.load(weight_ptr + col_offsets, mask=mask)
    output = (x_row * rstd) * weight

    tl.store(output_ptr + row_idx * n_cols + col_offsets, output, mask=mask)

def triton_rmsnorm(x, weight, eps=1e-06):
    batch_size, seq_len, hidden_size = x.shape
    n_rows = batch_size * seq_len

    x_flat = x.view(n_rows, hidden_size)
    output = torch.empty_like(x_flat)

    BLOCK_SIZE = triton.next_power_of_2(hidden_size)
    grid = (n_rows, )

    rmsnorm_kernel[grid](
        x_flat, weight, output,
        n_rows, hidden_size, eps,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return output.view(batch_size, seq_len, hidden_size)

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        return triton_rmsnorm(hidden_states, self.weight, self.variance_epsilon)

class Attention(nn.Module):
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.causal = True
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)

        self.register_buffer("cache_k", None, persistent=False)
        self.register_buffer("cache_v", None, persistent=False)

    def reset_cache(self):
        self.cache_k, self.cache_v = None, None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        use_cache=False
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Project to Q, K, V
        query_states = self.q_proj(hidden_states).view(batch_size, seq_len, self.config.num_attention_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(batch_size, seq_len, self.config.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(batch_size, seq_len, self.config.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if use_cache:
            if self.cache_k is None:
                self.cache_k, self.cache_v = key_states, value_states
            else:
                self.cache_k = torch.cat([self.cache_k, key_states], dim=2)
                self.cache_v = torch.cat([self.cache_v, value_states], dim=2)
            key_states, value_states = self.cache_k, self.cache_v

        attn_output, attn_weights = eager_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0,
            scaling=self.scaling,
        )

        # attn_output shape: [batch, num_heads, query_seq_len, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.config.num_attention_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.functional.silu

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

class DecoderLayer(nn.Module):
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Attention(config=config, layer_idx=layer_idx)
        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention_type = config.layer_types[layer_idx]

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = False,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

class Qwen2RotaryEmbedding(nn.Module):
    def __init__(self, config: Qwen2Config, device=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        dim = config.hidden_size // config.num_attention_heads
        base = getattr(config, 'rope_theta', 10000.0)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.attention_scaling = 1.0

        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1,2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

class Qwen2Model(nn.Module):
    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2RotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types

        self.current_pos = 0

    def reset_kv_cache(self):
        for layer in self.layers:
            layer.self_attn.reset_cache()
        self.current_pos = 0

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        input_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
    ) -> torch.Tensor:
        if input_embeds is None:
            input_embeds = self.embed_tokens(input_ids)

        if position_ids is None:
            if use_cache:
                pos_ids = torch.arange(
                    self.current_pos, self.current_pos + input_embeds.shape[1],
                    device=input_embeds.device, dtype=torch.long
                ).unsqueeze(0)
                self.current_pos += input_embeds.shape[1]
                position_ids = pos_ids
            else:
                position_ids = torch.arange(input_embeds.shape[1], device=input_embeds.device).unsqueeze(0)

        if use_cache and hasattr(self.layers[0].self_attn, 'cache_k') and self.layers[0].self_attn.cache_k is not None:
            query_len = input_embeds.shape[1]
            key_len = self.layers[0].self_attn.cache_k.shape[2] + query_len
            causal_mask = create_causal_mask(key_len, input_embeds.device, input_embeds.dtype)
            # Only use the part of the mask relevant to the current query
            causal_mask = causal_mask[:, :, -query_len:, :]
        else:
            seq_len = input_embeds.shape[1]
            causal_mask = create_causal_mask(seq_len, input_embeds.device, input_embeds.dtype)

        causal_mask_mapping = {
            "full_attention": causal_mask,
            "sliding_attention": causal_mask,
        }

        hidden_states = input_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                use_cache=use_cache
            )

        hidden_states = self.norm(hidden_states)
        return hidden_states

class Qwen2ForCausalLM(nn.Module):
    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.model = Qwen2Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None, position_ids=None, use_cache=False):
        hidden_states = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=use_cache
        )
        logits = self.lm_head(hidden_states)
        return logits

def generate_text(model, tokenizer, prompt, max_length=100, temperature=0.7, top_p=0.9, repetition_penalty=1.1):
    model.eval()
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    original_length = input_ids.shape[1]

    model.model.reset_kv_cache()
    generated_tokens = []

    with torch.no_grad():
        # First pass with full prompt
        logits = model(input_ids, use_cache=True)
        next_token_logits = logits[0, -1, :].clone()

        if generated_tokens:
            for token_id in set(generated_tokens):
                if next_token_logits[token_id] < 0:
                    next_token_logits[token_id] *= repetition_penalty
                else:
                    next_token_logits[token_id] /= repetition_penalty

        next_token_logits = next_token_logits / temperature

        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
        sorted_indices_to_remove[0] = False
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        next_token_logits[indices_to_remove] = float('-inf')

        probs = torch.softmax(next_token_logits, dim=-1)
        next_token_id = torch.multinomial(probs, num_samples=1)
        generated_tokens.append(next_token_id.item())

        if next_token_id.item() == tokenizer.eos_token_id:
            generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            return generated_text

        # Continue generating token by token
        for _ in range(max_length - 1):
            logits = model(next_token_id.unsqueeze(0), use_cache=True)
            next_token_logits = logits[0, -1, :].clone()

            if generated_tokens:
                for token_id in set(generated_tokens):
                    if next_token_logits[token_id] < 0:
                        next_token_logits[token_id] *= repetition_penalty
                    else:
                        next_token_logits[token_id] /= repetition_penalty

            next_token_logits = next_token_logits / temperature

            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = False
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            next_token_logits[indices_to_remove] = float('-inf')

            probs = torch.softmax(next_token_logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)
            generated_tokens.append(next_token_id.item())

            if next_token_id.item() == tokenizer.eos_token_id:
                break

    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return generated_text
