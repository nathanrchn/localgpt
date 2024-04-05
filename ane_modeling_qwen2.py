import torch
from torch import nn
from transformers import Qwen2Config
from argmaxtools import nn as ann, utils

class Qwen2RotaryEmbedding(nn.Module):
    def __init__(self, config: Qwen2Config) -> None:
        super().__init__()

class Qwen2RMSNorm(nn.Module):
    pass

class Qwen2Attention(nn.Module):
    def __init__(self, config: Qwen2Config) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads

        self.q_proj = nn.Conv2d(config.hidden_size, self.num_heads * self.head_dim, 1, bias=True)
        self.k_proj = nn.Conv2d(config.hidden_size, self.num_key_value_heads * self.head_dim, 1, bias=True)
        self.v_proj = nn.Conv2d(config.hidden_size, self.num_key_value_heads * self.head_dim, 1, bias=True)
        self.out_proj = nn.Conv2d(self.num_heads * self.head_dim, config.hidden_size, 1, bias=False)

        self.rotary_emb = Qwen2RotaryEmbedding(config)

class Qwen2MLP(nn.Module):
    def __init__(self, config: Qwen2Config) -> None:
        self.gate_proj = nn.Conv2d(config.hidden_size, config.intermediate_size, 1, bias=False)
        self.up_proj = nn.Conv2d(config.hidden_size, config.intermediate_size, 1, bias=False)
        self.down_proj = nn.Conv2d(config.intermediate_size, config.hidden_size, 1, bias=False)

class Qwen2DecoderLayer(nn.Module):
    def __init__(self, config: Qwen2Config) -> None:
        self.self_attn = Qwen2Attention(config)
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config)
        self.post_attention_layernorm = Qwen2RMSNorm(config)

class Qwen2Model(nn.Module):
    def __init__(self, config: Qwen2Config) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen2DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self._attn_implementation = config._attn_implementation
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

class Qwen2ForCausalLM(nn.Module):
    def __init__(self, config: Qwen2Config) -> None:
        super().__init__()
        self.model = Qwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Conv2d(config.hidden_size, config.vocab_size, 1, bias=False)
