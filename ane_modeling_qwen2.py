import torch
from typing import Tuple
from transformers import Qwen2Config
from argmaxtools import nn as ann, utils
from torch import nn, Tensor, functional as F

class Qwen2RMSNorm(nn.Module):
    def __init__(self, config: Qwen2Config) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.rms_norm_eps = config.rms_norm_eps
        self.weight = nn.Parameter(torch.ones(self.hidden_size))

    def forward(self, hidden_states: Tensor) -> Tensor:
        # hidden_states: (batch_size, hidden_size, 1, seq_len)
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(dim=1, keepdim=True) # var on hidden_size => dim is 1 instead of -1
        hidden_states *= torch.rsqrt(variance + self.rms_norm_eps)
        return self.weight.view(1, self.hidden_size, 1, 1) * hidden_states.to(input_dtype)

class Qwen2RotaryEmbedding(nn.Module):
    def __init__(self, config: Qwen2Config) -> None:
        super().__init__()
    
    def forward(self, query: Tensor, key: Tensor, position_ids: Tensor) -> Tuple[Tensor, Tensor]:
        # query: (batch_size, hidden_size, 1, seq_len)
        # key: (batch_size, hidden_size, 1, seq_len)
        # position_ids: (batch_size, seq_len)
        pass

class Qwen2Attention(nn.Module):
    def __init__(self, config: Qwen2Config) -> None:
        super().__init__()

    def forward(self, hidden_states: Tensor, attention_mask: Tensor, position_ids: Tensor, past_kv_values: Tensor) -> Tuple[Tensor, Tensor]:
        # hidden_states: (batch_size, hidden_size, 1, seq_len)
        # attention_mask: (batch_size, seq_len)
        # position_ids: (batch_size, seq_len)
        # past_kv_values: (2, batch_size, past_seq_len, hidden_size)
        pass

class Qwen2MLP(nn.Module):
    def __init__(self, config: Qwen2Config) -> None:
        super().__init__()
        self.gate_proj = nn.Conv2d(config.hidden_size, config.intermediate_size, 1, bias=False)
        self.up_proj = nn.Conv2d(config.hidden_size, config.intermediate_size, 1, bias=False)
        self.down_proj = nn.Conv2d(config.intermediate_size, config.hidden_size, 1, bias=False)

    def forward(self, hidden_states: Tensor) -> Tensor:
        # hidden_states: (batch_size, hidden_size, 1, seq_len)
        return self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))

class Qwen2DecoderLayer(nn.Module):
    def __init__(self, config: Qwen2Config) -> None:
        super().__init__()

    def forward(self, hidden_states: Tensor, attention_mask: Tensor, position_ids: Tensor, past_kv_values: Tensor) -> Tuple[Tensor, Tensor]:
        # hidden_states: (batch_size, hidden_size, 1, seq_len)
        # attention_mask: (batch_size, seq_len)
        # position_ids: (batch_size, seq_len)
        # past_kv_values: (2, batch_size, past_seq_len, hidden_size)
        pass

class Qwen2Model(nn.Module):
    def __init__(self, config: Qwen2Config) -> None:
        super().__init__()

    def forward(self, input_ids: Tensor, attention_mask: Tensor, position_ids: Tensor, past_kv_values: Tensor) -> Tuple[Tensor, Tensor]:
        # input_ids: (batch_size, seq_len)
        # attention_mask: (batch_size, seq_len)
        # position_ids: (batch_size, seq_len)
        # past_kv_values: (num_hidden_layers, 2, batch_size, past_seq_len, hidden_size)
        pass

class Qwen2ForCausalLM(nn.Module):
    def __init__(self, config: Qwen2Config) -> None:
        super().__init__()

    def forward(self, input_ids: Tensor, attention_mask: Tensor, position_ids: Tensor, past_kv_values: Tensor) -> Tuple[Tensor, Tensor]:
        # input_ids: (batch_size, seq_len)
        # attention_mask: (batch_size, seq_len)
        # position_ids: (batch_size, seq_len)
        # past_kv_values: (num_hidden_layers, 2, batch_size, past_seq_len, hidden_size)
        pass