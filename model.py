import torch
import torch.nn as nn
import math


# ----------------------------------------
# Input Embedding Layer
# ----------------------------------------
class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # x: (batch, seq_len)
        # output: (batch, seq_len, d_model)
        return self.embedding(x) * math.sqrt(self.d_model)


# ----------------------------------------
# Positional Encoding
# ----------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # pe shape: (1, seq_len, d_model)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ----------------------------------------
# Layer Normalization
# ----------------------------------------
class LayerNormalization(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1, 1, d_model))
        self.bias = nn.Parameter(torch.zeros(1, 1, d_model))

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


# ----------------------------------------
# Feed Forward Network
# ----------------------------------------
class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.linear_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


# ----------------------------------------
# Multi-Head Attention
# ----------------------------------------
class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, heads: int, dropout: float):
        super().__init__()
        assert d_model % heads == 0

        self.d_k = d_model // heads
        self.heads = heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout):
        """
        query: (batch, heads, seq_len_q, d_k)
        key:   (batch, heads, seq_len_k, d_k)
        value: (batch, heads, seq_len_v, d_k)
        """

        d_k = query.size(-1)

        # scores: (batch, heads, seq_len_q, seq_len_k)
        scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            # mask: (batch, 1, 1, seq_len_k) or broadcastable
            scores = scores.masked_fill(mask == 0, -1e9)

        # attention weights: (batch, heads, seq_len_q, seq_len_k)
        attention_weights = scores.softmax(dim=-1)

        if dropout is not None:
            attention_weights = dropout(attention_weights)

        # output: (batch, heads, seq_len_q, d_k)
        output = attention_weights @ value

        return output, attention_weights

    def forward(self, q, k, v, mask=None):
        """
        q, k, v: (batch, seq_len, d_model)
        """

        batch_size = q.size(0)

        # Linear projections
        # (batch, seq_len, d_model)
        query = self.w_q(q)
        key   = self.w_k(k)
        value = self.w_v(v)

        # Split into heads
        # (batch, seq_len, heads, d_k)
        query = query.view(batch_size, -1, self.heads, self.d_k)
        key   = key.view(batch_size, -1, self.heads, self.d_k)
        value = value.view(batch_size, -1, self.heads, self.d_k)

        # Transpose for attention
        # (batch, heads, seq_len, d_k)
        query = query.transpose(1, 2)
        key   = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # Apply attention
        # x: (batch, heads, seq_len, d_k)
        x, self.attention_scores = self.attention(
            query, key, value, mask, self.dropout
        )

        # Concatenate heads
        # (batch, seq_len, heads, d_k)
        x = x.transpose(1, 2)

        # (batch, seq_len, d_model)
        x = x.contiguous().view(batch_size, -1, self.heads * self.d_k)

        # Final projection
        return self.w_o(x)


# ----------------------------------------
# Residual Connection
# ----------------------------------------
class ResidualConnection(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.norm = LayerNormalization(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        # x: (batch, seq_len, d_model)
        return x + self.dropout(sublayer(self.norm(x)))


# ----------------------------------------
# Encoder Block
# ----------------------------------------
class EncoderBlock(nn.Module):
    def __init__(
        self,
        self_attention_block: MultiHeadAttentionBlock,
        feed_forward_block: FeedForwardBlock,
        d_model: int,
        dropout: float,
    ):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block

        self.residual_connections = nn.ModuleList(
            [ResidualConnection(d_model, dropout) for _ in range(2)]
        )

    def forward(self, x, src_mask):
        # Self-attention + residual
        x = self.residual_connections[0](
            x, lambda x: self.self_attention_block(x, x, x, src_mask)
        )

        # Feed-forward + residual
        x = self.residual_connections[1](x, self.feed_forward_block)

        return x


# ----------------------------------------
# Encoder Stack
# ----------------------------------------
class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList, d_model: int):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(d_model)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
