import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

class ESU(nn.Module):
    """
    Extended Sequence Unifier (ESU) for long-term interest extraction using
    multi-head cross-attention.

    输入:
        query_emb: Tensor of shape (B, D)           -- 正样本编码
        longtime_emb: Tensor of shape (B, L, D)     -- 长期序列编码
        lt_lengths: Tensor of shape (B,)            -- 各序列有效长度

    输出:
        Tensor of shape (B, D)                     -- 长期兴趣向量
    """
    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        # PyTorch's MultiheadAttention expects (L, B, E)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                query_emb: torch.Tensor,
                longtime_emb: torch.Tensor,
                lt_lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            query_emb:   (B, D)
            longtime_emb: (B, L, D)
            lt_lengths:   (B,)
        Returns:
            long_interest: (B, D)
        """
        B, L, D = longtime_emb.size()
        # Expand query to (B, 1, D)
        q = query_emb.unsqueeze(1)  # (B, 1, D)

        # Create key_padding_mask: True where padding
        # lt_lengths is number of valid tokens per batch element
        idx = torch.arange(L, device=lt_lengths.device).unsqueeze(0).expand(B, -1)  # (B, L)
        key_padding_mask = idx >= lt_lengths.unsqueeze(1)  # True for pads

        # Apply multi-head cross-attention: Q=q, K=V=longtime_emb
        # Attn returns (B, 1, D)
        attn_output, _ = self.attn(query=q, key=longtime_emb, value=longtime_emb,
                                   key_padding_mask=key_padding_mask)
        # Residual + LayerNorm
        out = self.layer_norm(query_emb + self.dropout(attn_output.squeeze(1)))
        return out