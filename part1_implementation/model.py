import torch
from torch import nn
import math


# Embedding Layer (Token + Learned Positional Embeddings)
class Embeddings(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_len, pad_token=0, dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=pad_token
        )
        self.position_embedding = nn.Embedding(max_len, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids):
        batch_size, seq_len = input_ids.size()
        positions = torch.arange(
            seq_len, device=input_ids.device
        ).unsqueeze(0).expand(batch_size, seq_len)

        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.layer_norm(x)
        return self.dropout(x)


# Multi-Head Self-Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, \
            "embed_dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        B, S, E = x.shape

        qkv = self.qkv(x).view(B, S, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        Q, K, V = qkv[0], qkv[1], qkv[2]

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask[:, None, None, :], float("-inf")
            )

        attn = self.attn_dropout(scores.softmax(dim=-1))
        out = (attn @ V).transpose(1, 2).contiguous().view(B, S, E)

        return self.out_proj(out), attn


# Transformer Encoder Layer
class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_hidden_dim, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden_dim, embed_dim),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        attn_out, attn = self.self_attn(x, key_padding_mask)
        x = self.norm1(x + self.dropout(attn_out))

        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))

        return x, attn


# Transformer Encoder Classifier
class TransformerEncoder(nn.Module):
    def __init__(self,
                 num_layers,
                 embed_dim,
                 num_heads,
                 ff_hidden_dim,
                 vocab_size,
                 max_len,
                 num_classes,
                 dropout=0.1,
                 pad_token=0):
        super().__init__()

        self.embeddings = Embeddings(
            vocab_size, embed_dim, max_len, pad_token, dropout
        )

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                embed_dim, num_heads, ff_hidden_dim, dropout
            )
            for _ in range(num_layers)
        ])

        self.classifier = nn.Linear(embed_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask=None):
        """
        attention_mask: True for padding positions
        """
        x = self.embeddings(input_ids)

        attn_weights = []
        for layer in self.layers:
            x, attn = layer(x, attention_mask)
            attn_weights.append(attn)

        # Mean pooling over non-padding tokens
        if attention_mask is not None:
            valid_mask = (~attention_mask).unsqueeze(-1)
            pooled = (x * valid_mask).sum(dim=1) / valid_mask.sum(dim=1)
        else:
            pooled = x.mean(dim=1)

        logits = self.classifier(self.dropout(pooled))
        return logits, attn_weights


# Model Factory
def create_model_instance(config, device):
    mcfg = config["model"]

    encoder = TransformerEncoder(
        num_layers=int(mcfg["num_layers"]),
        embed_dim=int(mcfg["embed_dim"]),
        num_heads=int(mcfg["num_heads"]),
        ff_hidden_dim=int(mcfg["dim_feedforward"]),
        vocab_size=int(mcfg["vocab_size"]),
        max_len=int(mcfg["max_len"]),
        num_classes=int(mcfg["num_classes"]),
        dropout=float(mcfg["dropout"]),
        pad_token=0
    )

    return encoder.to(device)
