import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence

from layers.sequence import DynamicGRU


def extract_last_state(packed: PackedSequence, lengths: torch.Tensor) -> torch.Tensor:
    padded, _ = pad_packed_sequence(packed, batch_first=True)
    batch_idx = torch.arange(padded.size(0), device=lengths.device)
    time_idx = lengths.clamp(min=1) - 1
    return padded[batch_idx, time_idx, :]


class ShortTermInterestExtractor(nn.Module):
    """
    短期交互兴趣提取器，仅使用 AUGRU

    输入:
        x: Tensor of shape (B, H, D)
        pos_query: Tensor of shape (B, D)
        lengths: Tensor of shape (B,)
    输出: interest Tensor of shape (B, D)
    """

    def __init__(self,
                 input_size: int = 16,
                 hidden_size: int = 16,
                 num_layers: int = 1,
                 dropout: float = 0.0,
                 bidirectional: bool = False):
        super().__init__()
        self.hidden_size = hidden_size
        # 单层 GRU
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )
        # AUGRU 动态更新门 GRU
        self.augru = DynamicGRU(input_size=hidden_size,
                                hidden_size=hidden_size)

    def forward(self,
                x: torch.Tensor,
                pos_query: torch.Tensor,
                lengths: torch.Tensor) -> torch.Tensor:
        # x: (B, H, D), pos_query: (B, D), lengths: (B,)
        # print(x.shape)
        # print(pos_query.shape)
        # print(lengths.shape)
        B, H, D = x.size()
        # 1. pack & GRU
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.gru(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=H)
        # 2. attention scores: dot(query, keys)
        q = pos_query.unsqueeze(1)  # (B,1,D)
        scores = torch.bmm(q, out.transpose(1,2)).squeeze(1)  # (B,H)
        att_scores = torch.softmax(scores, dim=1).unsqueeze(-1)  # (B,H,1)
        # 3. pack att_scores
        packed_x   = pack_padded_sequence(out, lengths.cpu(),   batch_first=True, enforce_sorted=False)
        packed_att = pack_padded_sequence(att_scores, lengths.cpu(), batch_first=True, enforce_sorted=False)
        # 4. AUGRU
        packed_h = self.augru(packed_x, packed_att)
        # 5. extract last state
        interest = extract_last_state(packed_h, lengths)
        return interest