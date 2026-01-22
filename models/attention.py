import torch
import torch.nn as nn
import math


class SelfAttention1D(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout=0.0):
        super(SelfAttention1D, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout,
                                               batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        Args:
            x (Tensor): shape (batch, channels, seq_len)
        """
        x_seq = x.transpose(1, 2)  # (batch, seq_len, channels)
        attn_out, _ = self.attention(x_seq, x_seq, x_seq)
        attn_out = self.norm(attn_out + x_seq)
        return attn_out.transpose(1, 2)


class SelfAttention2D(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout=0.0):
        super(SelfAttention2D, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout,
                                               batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        Args:
            x (Tensor): shape (batch, channels, height, width)
        """
        batch, channels, height, width = x.shape
        # Reshape to (batch, height*width, channels) for attention
        x_seq = x.view(batch, channels, height * width).transpose(1, 2)  # (batch, height*width, channels)
        attn_out, _ = self.attention(x_seq, x_seq, x_seq)
        attn_out = self.norm(attn_out + x_seq)
        # Reshape back to (batch, channels, height, width)
        attn_out = attn_out.transpose(1, 2).view(batch, channels, height, width)
        return attn_out

class LogitBiasedSelfAttention1D(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout=0.0):
        super().__init__()

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # QKV projections
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # SQI → attention-logit bias (lightweight, learnable)
        self.sqi_proj = nn.Conv1d(1, 1, kernel_size=3, padding=1)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, sqi):
        """
        Args:
            x   : Tensor (B, C, T)   – CNN feature map
            sqi : Tensor (B, T)      – SQI aligned to CNN time axis
        Returns:
            Tensor (B, C, T)
        """

        B, C, T = x.shape

        # (B, T, C)
        x_seq = x.transpose(1, 2)

        # ---- QKV ----
        qkv = self.qkv(x_seq)                      # (B, T, 3C)
        q, k, v = qkv.chunk(3, dim=-1)

        # (B, H, T, D)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # ---- Attention logits ----
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        # shape: (B, H, T, T)

        # ---- SQI logit bias ----
        # sqi: (B, T) → (B, 1, T)
        sqi = sqi.unsqueeze(1)

        # learnable smoothing + projection
        sqi_bias = self.sqi_proj(sqi).squeeze(1)   # (B, T)

        # add bias to *keys* dimension
        attn_logits = attn_logits + sqi_bias.unsqueeze(1).unsqueeze(2)

        # ---- Attention ----
        attn = torch.softmax(attn_logits, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)               # (B, H, T, D)

        # ---- Merge heads ----
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        out = self.out_proj(out)

        # ---- Residual + Norm ----
        out = self.norm(out + x_seq)

        return out.transpose(1, 2)                 # (B, C, T)
