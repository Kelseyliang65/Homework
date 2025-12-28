#!/usr/bin/env python3
"""
Plan 2: Transformer NMT with RMSNorm & Relative Positional Encoding (Fixed Mask Broadcasting)
Features:
1. RMSNorm for training stability
2. Relative Positional Encoding (RPE) for better generalization on long sequences
3. Explicit hooks (encode/decode/generator) for Beam Search
4. Fixed: Correct mask dimensions for multi-head attention
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        # x: (batch, seq_len, dim)
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.scale * x / rms


class RelativePositionalEncoding(nn.Module):
    """相对位置编码生成器"""

    def __init__(self, nhead, max_len=512):
        super().__init__()
        self.nhead = nhead
        self.max_len = max_len
        # (2*max_len+1) 个位置的嵌入向量
        self.rel_bias = nn.Embedding(2 * max_len + 1, nhead)

    def forward(self, seq_len_q, seq_len_k, device):
        # 生成相对位置矩阵
        pos_q = torch.arange(seq_len_q, device=device).unsqueeze(1)
        pos_k = torch.arange(seq_len_k, device=device).unsqueeze(0)
        rel_pos = pos_q - pos_k + self.max_len
        rel_pos = rel_pos.clamp(0, 2 * self.max_len)

        # (L_q, L_k, nhead) -> (nhead, L_q, L_k)
        return self.rel_bias(rel_pos).permute(2, 0, 1)


class AbsolutePositionalEncoding(nn.Module):
    """Sinusoidal absolute positional encoding."""

    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        return self.pe[:, :x.size(1), :]


class RelativeMultiheadAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, attn_mask=None, padding_mask=None, rel_pos_bias=None):
        bsz, q_len, _ = q.size()
        k_len = k.size(1)

        # 1. 投影并分头 -> (bsz, nhead, L, d_k)
        query = self.q_proj(q).view(bsz, q_len, self.nhead, self.d_k).transpose(1, 2)
        key = self.k_proj(k).view(bsz, k_len, self.nhead, self.d_k).transpose(1, 2)
        value = self.v_proj(v).view(bsz, k_len, self.nhead, self.d_k).transpose(1, 2)

        # 2. 计算 Attention Scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 3. 注入相对位置偏置 (RPE)
        if rel_pos_bias is not None:
            # rel_pos_bias: (nhead, q_len, k_len)
            # scores: (bsz, nhead, q_len, k_len)
            scores = scores + rel_pos_bias.unsqueeze(0)

        # 4. Masking
        # Padding Mask (bsz, 1, 1, k_len) or (bsz, 1, q_len, k_len)
        if padding_mask is not None:
            scores = scores.masked_fill(padding_mask == 0, float('-inf'))

        # Causal Mask (bsz, 1, L, L) or (1, 1, L, L)
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, float('-inf'))

        # 5. Softmax
        attn_probs = F.softmax(scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # 6. Weighted Sum
        out = torch.matmul(attn_probs, value)
        out = out.transpose(1, 2).contiguous().view(bsz, q_len, self.d_model)
        return self.out_proj(out)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout, use_rmsnorm=True):
        super().__init__()
        self.self_attn = RelativeMultiheadAttention(d_model, nhead, dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout)
        )
        Norm = RMSNorm if use_rmsnorm else nn.LayerNorm
        self.norm1 = Norm(d_model)
        self.norm2 = Norm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask, rel_pos_bias):
        # Pre-Norm Architecture
        nx = self.norm1(x)
        x = x + self.dropout(self.self_attn(nx, nx, nx, padding_mask=src_mask, rel_pos_bias=rel_pos_bias))
        nx = self.norm2(x)
        x = x + self.dropout(self.ff(nx))
        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout, use_rmsnorm=True):
        super().__init__()
        self.self_attn = RelativeMultiheadAttention(d_model, nhead, dropout)
        self.cross_attn = RelativeMultiheadAttention(d_model, nhead, dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout)
        )
        Norm = RMSNorm if use_rmsnorm else nn.LayerNorm
        self.norm1 = Norm(d_model)
        self.norm2 = Norm(d_model)
        self.norm3 = Norm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, tgt_mask, mem_mask, rel_pos_bias):
        # 1. Self Attention (with RPE)
        nx = self.norm1(x)
        x = x + self.dropout(self.self_attn(nx, nx, nx, attn_mask=tgt_mask, rel_pos_bias=rel_pos_bias))

        # 2. Cross Attention (Standard, usually no RPE here)
        nx = self.norm2(x)
        x = x + self.dropout(self.cross_attn(nx, memory, memory, padding_mask=mem_mask))

        # 3. Feed Forward
        nx = self.norm3(x)
        x = x + self.dropout(self.ff(nx))
        return x


class TransformerNMT(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=512, nhead=8, n_enc=3, n_dec=3,
                 d_ff=2048, dropout=0.3, max_len=128, use_rmsnorm=True, pos_encoding="relative"):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead

        self.src_embed = nn.Embedding(src_vocab, d_model, padding_idx=0)
        self.tgt_embed = nn.Embedding(tgt_vocab, d_model, padding_idx=0)
        self.pos_encoding = pos_encoding
        if pos_encoding == "relative":
            self.pos_enc = RelativePositionalEncoding(nhead, max_len)
        elif pos_encoding == "absolute":
            self.pos_enc = AbsolutePositionalEncoding(d_model, max_len)
        else:
            raise ValueError(f"Unknown pos_encoding: {pos_encoding}")
        self.dropout = nn.Dropout(dropout)

        self.enc_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, d_ff, dropout, use_rmsnorm)
            for _ in range(n_enc)
        ])
        self.dec_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, nhead, d_ff, dropout, use_rmsnorm)
            for _ in range(n_dec)
        ])

        self.out_proj = nn.Linear(d_model, tgt_vocab)

        # 初始化参数
        for p in self.parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p)

    def make_std_mask(self, tgt, pad_id=0):
        """生成 Target 的标准掩码 (Causal Mask & Padding Mask)"""
        tgt_mask = (tgt != pad_id).unsqueeze(-2)  # (B, 1, L)
        tgt_mask = tgt_mask & torch.autograd.Variable(
            self.subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        # 【关键修复】扩展维度以匹配多头注意力 (B, L, L) -> (B, 1, L, L)
        return tgt_mask.unsqueeze(1)

    @staticmethod
    def subsequent_mask(size):
        """生成三角掩码"""
        attn_shape = (1, size, size)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
        return subsequent_mask == 0

    def encode(self, src, src_mask=None):
        """
        Encoder 前向传播
        """
        batch_size, seq_len = src.size()
        rel_pos_bias = None
        if self.pos_encoding == "relative":
            rel_pos_bias = self.pos_enc(seq_len, seq_len, src.device)  # (nhead, L, L)

        # 兼容 inference.py 传过来的 src_mask 形状
        if src_mask is None:
            src_mask = (src != 0).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, L)

        x = self.src_embed(src) * math.sqrt(self.d_model)
        if self.pos_encoding == "absolute":
            x = x + self.pos_enc(x)
        x = self.dropout(x)

        for layer in self.enc_layers:
            x = layer(x, src_mask, rel_pos_bias)

        return x

    def decode(self, tgt, memory, src_mask, tgt_mask):
        """
        Decoder 前向传播
        """
        batch_size, seq_len = tgt.size()
        rel_pos_bias = None
        if self.pos_encoding == "relative":
            rel_pos_bias = self.pos_enc(seq_len, seq_len, tgt.device)

        x = self.tgt_embed(tgt) * math.sqrt(self.d_model)
        if self.pos_encoding == "absolute":
            x = x + self.pos_enc(x)
        x = self.dropout(x)

        for layer in self.dec_layers:
            x = layer(x, memory, tgt_mask, src_mask, rel_pos_bias)

        return x

    def generator(self, x):
        """Beam Search 需要调用的生成层"""
        return self.out_proj(x)

    def forward(self, src, tgt):
        """
        训练时的标准 Forward
        src: (B, S)
        tgt: (B, T)
        """
        # 构造 Masks
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, S)
        tgt_mask = self.make_std_mask(tgt, 0)  # (B, 1, T, T)

        # Encode
        memory = self.encode(src, src_mask)

        # Decode
        out = self.decode(tgt, memory, src_mask, tgt_mask)

        # Generator
        return self.generator(out)

    @torch.no_grad()
    def translate(self, src, bos_id, eos_id, max_len=100):
        """
        Greedy Search (Fallback if Beam Search fails)
        """
        self.eval()
        bsz = src.size(0)
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        memory = self.encode(src, src_mask)

        ys = torch.full((bsz, 1), bos_id, dtype=torch.long, device=src.device)
        finished = torch.zeros(bsz, dtype=torch.bool, device=src.device)

        for _ in range(max_len):
            tgt_mask = self.make_std_mask(ys, 0)
            out = self.decode(ys, memory, src_mask, tgt_mask)
            prob = self.generator(out[:, -1])
            next_tok = prob.argmax(-1, keepdim=True)

            # Mask logic
            next_tok = next_tok.masked_fill(finished.unsqueeze(1), 0)
            ys = torch.cat([ys, next_tok], dim=1)

            finished |= (next_tok.squeeze(1) == eos_id)
            if finished.all(): break

        return ys