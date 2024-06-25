import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Optional
from torch import Tensor


class PositionalEncoding(nn.Module):
    def __init__(self, max_seq_len: int, embd_dim: int) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.embd_dim = embd_dim
        pe = torch.zeros(max_seq_len, embd_dim)

        position = torch.arange(0, max_seq_len).unsqueeze(1)
        power = torch.arange(0, embd_dim, 2) / embd_dim
        pe[:, 0::2] = torch.sin(position / (10000**power))
        pe[:, 1::2] = torch.cos(position / (10000**power))
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x: Tensor, padding_masks: Optional[Tensor] = None) -> Tensor:
        B, T, E = x.shape
        x = x + self.pe[:, :T, :]
        if padding_masks is not None:
            padding_masks = padding_masks.unsqueeze(-1)
            x = x.masked_fill(padding_masks==0, 0)
        return x


class LayerNorm(nn.Module):

    def __init__(self, ndim: int, bias: bool = True, eps: float = 1e-5) -> None:
        super().__init__()
        self.ndim = ndim
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        sd = x.std(dim=-1, keepdim=True)
        output = (x - mean) / (sd + self.eps) * self.weight
        if self.bias is not None:
            output = output + self.bias
        return output



class MultiHeadAttention(nn.Module):

    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        causal: bool = False,
    ) -> None:
        
        super().__init__()

        # Note: in the "Attention Is All You Need" paper, key_dim = value_dim = (d_model / n_heads)
        assert d_model % n_heads == 0

        self.n_heads = n_heads
        self.key_dim = d_model // n_heads
        self.value_dim = self.key_dim

        self.W_q = nn.Linear(d_model, n_heads * self.key_dim, bias=False)
        self.W_k = nn.Linear(d_model, n_heads * self.key_dim, bias=False)
        self.W_v = nn.Linear(d_model, n_heads * self.value_dim, bias=False)
        self.W_o = nn.Linear(n_heads * self.value_dim, d_model, bias=False)

        self.causal = causal


    def forward(
        self, 
        query: Tensor, 
        key: Tensor, 
        value: Tensor, 
        query_padding_masks: Optional[Tensor] = None, 
        key_padding_masks: Optional[Tensor] = None
    ) -> Tensor:
        
        # B: batch size
        # H: number of heads
        # T_q, T_k: sequence length of query and key
        # E: embedding size

        B, T_q, E = query.shape
        _, T_k, _ = key.shape
        
        q = self.W_q(query)  # (B, T_q, H * key_dim)
        k = self.W_k(key)    # (B, T_k, H * key_dim)
        v = self.W_v(value)  # (B, T_k, H * value_dim)

        # Split out heads dimension
        q = q.view(B, T_q, self.n_heads, self.key_dim)    # (B, T_q, H, key_dim)
        k = k.view(B, T_k, self.n_heads, self.key_dim)    # (B, T_k, H, key_dim)
        v = v.view(B, T_k, self.n_heads, self.value_dim)  # (B, T_k, H, value_dim)

        # Move the heads dimension to the second position
        q = q.transpose(1, 2)   # (B, H, T_q, key_dim)
        k = k.transpose(1, 2)   # (B, H, T_k, key_dim)
        v = v.transpose(1, 2)   # (B, H, T_k, value_dim)

        # (B, H, T_q, key_dim) x (B, H, key_dim, T_k) -> (B, H, T_q, T_k)
        attention = (q @ k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.key_dim))


        mask = torch.ones((B, 1, T_q, T_k), dtype=int, device=self.W_q.weight.device)
        if self.causal:
            mask = torch.tril(mask)
        if key_padding_masks is not None:
            mask = mask * key_padding_masks.view(B, 1, 1, T_k)

        attention = attention.masked_fill(mask == 0, float('-inf'))

        attention = F.softmax(attention, dim=-1)

        if query_padding_masks is not None:
            attention = attention * query_padding_masks.view(B, 1, T_q, 1)

        # (B, H, T_q, T_k) x (B, H, T_k, value_dim) -> (B, H, T_q, value_dim)
        attention = attention @ v

        # Concatenate heads:
        #     1. Move heads dimension to the second to last dim
        #        (B, H, T_q, value_dim) -> (B, T_q, H, value_dim)
        #     2. Concatenate heads
        #        (B, T_q, H, value_dim) -> (B, T_q, H * value_dim)

        attention = attention.transpose(1, 2).contiguous().view(B, T_q, self.n_heads * self.value_dim)

        # (B, T_q, H * value_dim) x (H * value_dim, d_model) -> (B, T_q, d_model)
        output = self.W_o(attention)

        return output


class FeedForward(nn.Module):

    def __init__(
        self, 
        d_model: int = 512, 
        hidden_dim: int = 2048, 
        bias: bool = True
    ) -> None:
        
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim, bias=bias),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model, bias=bias),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)



class ResidualBlock(nn.Module):
    def __init__(
        self, 
        dropout: float = 0.1, 
        d_model: int = 512, 
        bias: bool = True, 
        layer_norm_eps: float = 1e-5
    ) -> None:
        
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNorm(ndim=d_model, bias=bias, eps=layer_norm_eps)


    def forward(self, module_input: Tensor, module_output: Tensor) -> Tensor:
        output = self.dropout(module_output)
        output = output + module_input
        output = self.layer_norm(output)
        return output



class EncoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        feedforward_dim: int = 2048,
        dropout: float = 0.1,
        bias: bool = True,
        layer_norm_eps: float = 10**-5
    ) -> None:
        
        super().__init__()
        self.self_attention = MultiHeadAttention(
            d_model,
            n_heads,
            causal=False,
        )
        self.feed_forward = FeedForward(d_model, feedforward_dim, bias=bias)
        self.residual_block = ResidualBlock(
            dropout, d_model, bias, layer_norm_eps
        )


    def forward(
        self, 
        encoder_input: Tensor, 
        source_padding_masks: Optional[Tensor] = None
    ) -> Tensor:
        
        att_output = self.self_attention(
            query=encoder_input, 
            key=encoder_input, 
            value=encoder_input, 
            query_padding_masks=source_padding_masks, 
            key_padding_masks=source_padding_masks
        )
        att_output = self.residual_block(encoder_input, att_output)

        ff_output = self.feed_forward(att_output)
        output = self.residual_block(att_output, ff_output)
        return output


class DecoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        feedforward_dim: int = 2048,
        dropout: float = 0.1,
        bias: bool = True,
        layer_norm_eps: float = 10**-5
    ) -> None:
        
        super().__init__()
        self.causal_self_attention = MultiHeadAttention(
            d_model,
            n_heads,
            causal=True,
        )

        self.cross_attention = MultiHeadAttention(
            d_model,
            n_heads,
            causal=False,
        )

        self.feed_forward = FeedForward(d_model, feedforward_dim)
        self.residual_block = ResidualBlock(
            dropout, 
            d_model, 
            bias, 
            layer_norm_eps
        )


    def forward(
        self, 
        decoder_input: Tensor, 
        encoder_output: Tensor, 
        source_padding_masks: Optional[Tensor] = None, 
        target_padding_masks: Optional[Tensor] = None
    ) -> Tensor:
        
        self_att_output = self.causal_self_attention(
            query=decoder_input, 
            key=decoder_input, 
            value=decoder_input, 
            query_padding_masks=target_padding_masks, 
            key_padding_masks=target_padding_masks
        )
        self_att_output = self.residual_block(decoder_input, self_att_output)

        cross_att_output = self.cross_attention(
            query=self_att_output, 
            key=encoder_output, 
            value=encoder_output,
            query_padding_masks=target_padding_masks,
            key_padding_masks=source_padding_masks
        )
        cross_att_output = self.residual_block(self_att_output, cross_att_output)

        ff_output = self.feed_forward(cross_att_output)
        output = self.residual_block(cross_att_output, ff_output)

        return output
