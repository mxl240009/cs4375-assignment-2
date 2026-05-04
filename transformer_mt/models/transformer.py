import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# --- Attention ---
# core formula: softmax(QK^T / sqrt(d_k)) * V

class SelfAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        d_k = Q.size(-1)

        # step 1: dot product Q and K, scale down
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

        # step 2: mask out future positions (used in decoder)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # step 3: softmax to get attention weights
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # step 4: weighted sum of values
        out = torch.matmul(attn, V)
        return out, attn


# --- Multi Head Attention ---
# split into h heads, run attention separately, then concat

class MultiHeadAttn(nn.Module):
    def __init__(self, d_model, h, dropout=0.1):
        super().__init__()
        assert d_model % h == 0
        self.h = h
        self.d_k = d_model // h

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)

        self.attn = SelfAttention(dropout)

    def split(self, x):
        # (B, T, d_model) -> (B, h, T, d_k)
        B, T, _ = x.size()
        x = x.view(B, T, self.h, self.d_k)
        return x.transpose(1, 2)

    def forward(self, Q, K, V, mask=None):
        Q = self.split(self.wq(Q))
        K = self.split(self.wk(K))
        V = self.split(self.wv(V))

        out, weights = self.attn(Q, K, V, mask)

        # concat heads back: (B, h, T, d_k) -> (B, T, d_model)
        B, _, T, _ = out.size()
        out = out.transpose(1, 2).contiguous().view(B, T, -1)

        return self.wo(out), weights


# --- Feed Forward ---
# two linear layers with relu in the middle
# d_ff is usually 4x d_model

class FFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.l1 = nn.Linear(d_model, d_ff)
        self.l2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.l2(self.drop(F.relu(self.l1(x))))


# --- Positional Encoding ---
# adds position info to embeddings since transformer has no recurrence
# using fixed sin/cos pattern from the paper

class PosEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(pos * div)  # even
        pe[:, 1::2] = torch.cos(pos * div)  # odd

        pe = pe.unsqueeze(0)  # add batch dim
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# --- Encoder Layer ---
# one block: self attention + ffn
# each sublayer has residual connection + layernorm

class EncoderBlock(nn.Module):
    def __init__(self, d_model, h, d_ff, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttn(d_model, h, dropout)
        self.ffn  = FFN(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # sublayer 1
        attn_out, _ = self.attn(x, x, x, mask)
        x = self.norm1(x + self.drop(attn_out))
        # sublayer 2
        x = self.norm2(x + self.drop(self.ffn(x)))
        return x


# --- Decoder Layer ---
# one block: masked self attn + cross attn + ffn

class DecoderBlock(nn.Module):
    def __init__(self, d_model, h, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn  = MultiHeadAttn(d_model, h, dropout)
        self.cross_attn = MultiHeadAttn(d_model, h, dropout)
        self.ffn        = FFN(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        # sublayer 1: masked self attention on target
        a1, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.drop(a1))

        # sublayer 2: cross attention - Q from decoder, K/V from encoder
        a2, _ = self.cross_attn(x, enc_out, enc_out, src_mask)
        x = self.norm2(x + self.drop(a2))

        # sublayer 3: ffn
        x = self.norm3(x + self.drop(self.ffn(x)))
        return x


# --- Encoder ---

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, h, d_ff, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pe    = PosEncoding(d_model, dropout=dropout)
        self.blocks = nn.ModuleList([EncoderBlock(d_model, h, d_ff, dropout) for _ in range(N)])
        self.scale  = math.sqrt(d_model)

    def forward(self, src, mask=None):
        x = self.pe(self.embed(src) * self.scale)
        for block in self.blocks:
            x = block(x, mask)
        return x


# --- Decoder ---

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, h, d_ff, dropout=0.1):
        super().__init__()
        self.embed  = nn.Embedding(vocab_size, d_model)
        self.pe     = PosEncoding(d_model, dropout=dropout)
        self.blocks = nn.ModuleList([DecoderBlock(d_model, h, d_ff, dropout) for _ in range(N)])
        self.fc     = nn.Linear(d_model, vocab_size)
        self.scale  = math.sqrt(d_model)

    def forward(self, tgt, enc_out, src_mask=None, tgt_mask=None):
        x = self.pe(self.embed(tgt) * self.scale)
        for block in self.blocks:
            x = block(x, enc_out, src_mask, tgt_mask)
        return self.fc(x)


# --- Full Model ---

class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=256, h=8, N=3, d_ff=512, dropout=0.1):
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, N, h, d_ff, dropout)
        self.decoder = Decoder(tgt_vocab, d_model, N, h, d_ff, dropout)
        self._init_params()

    def _init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def src_mask(self, src, pad_idx):
        return (src != pad_idx).unsqueeze(1).unsqueeze(2)

    def tgt_mask(self, tgt, pad_idx):
        T = tgt.size(1)
        pad_mask    = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)
        causal_mask = torch.tril(torch.ones(T, T, device=tgt.device)).bool()
        return pad_mask & causal_mask

    def forward(self, src, tgt, src_pad, tgt_pad):
        src_m = self.src_mask(src, src_pad)
        tgt_m = self.tgt_mask(tgt, tgt_pad)
        enc   = self.encoder(src, src_m)
        out   = self.decoder(tgt, enc, src_m, tgt_m)
        return out