import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence

class AUGRUCell(nn.Module):
    """
    Attentional Update Gate GRU Cell (AUGRU)

    在标准 GRU 的基础上，用 att_score 加权更新门:
        z_t = sigmoid(W_z x_t + U_z h_{t-1})
        \hat z_t = att_score * z_t
        h_t = (1 - \hat z_t) * h_{t-1} + \hat z_t * \tilde h_t
    """
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # 权重矩阵
        self.weight_ih = nn.Parameter(torch.Tensor(3 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.Tensor(3 * hidden_size, hidden_size))
        # 偏置
        if bias:
            self.bias_ih = nn.Parameter(torch.zeros(3 * hidden_size))
            self.bias_hh = nn.Parameter(torch.zeros(3 * hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight_ih)
        nn.init.orthogonal_(self.weight_hh)
        if self.bias_ih is not None:
            nn.init.zeros_(self.bias_ih)
            nn.init.zeros_(self.bias_hh)

    def forward(self, x: torch.Tensor, hx: torch.Tensor, att_score: torch.Tensor) -> torch.Tensor:
        # x: (batch, input_size)
        # hx: (batch, hidden_size)
        # att_score: (batch,) or (batch,1)
        gi = F.linear(x, self.weight_ih, self.bias_ih)    # (batch, 3*hidden)
        gh = F.linear(hx, self.weight_hh, self.bias_hh)  # (batch, 3*hidden)
        i_r, i_z, i_n = gi.chunk(3, dim=1)
        h_r, h_z, h_n = gh.chunk(3, dim=1)
        # 重置门
        reset_gate = torch.sigmoid(i_r + h_r)
        # 原始更新门
        upd_gate = torch.sigmoid(i_z + h_z)
        # 候选隐藏状态
        new_state = torch.tanh(i_n + reset_gate * h_n)
        # 应用注意力权重到更新门
        att = att_score.unsqueeze(1) if att_score.dim() == 1 else att_score
        upd_gate = att * upd_gate
        # 最终隐藏状态
        hy = (1 - upd_gate) * hx + upd_gate * new_state
        return hy

class DynamicGRU(nn.Module):
    """
    Dynamic GRU 仅含 AUGRU，通过 PackedSequence 接收输入和对应的注意力分数序列
    输入:
        inputs: PackedSequence of shape [sum(batch_sizes), input_size]
        att_scores: PackedSequence of shape [sum(batch_sizes), 1]
    输出:
        PackedSequence of hidden states (sum(batch_sizes), hidden_size)
    """
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True):
        super().__init__()
        self.cell = AUGRUCell(input_size, hidden_size, bias)

    def forward(self, inputs: PackedSequence, att_scores: PackedSequence, hx: torch.Tensor = None) -> PackedSequence:
        data, batch_sizes, sorted_idx, unsorted_idx = inputs
        att_data, _, _, _ = att_scores
        max_batch = int(batch_sizes[0])
        if hx is None:
            hx = data.new_zeros(max_batch, self.cell.hidden_size)

        outputs = []
        offset = 0
        h = hx
        for batch in batch_sizes:
            x_batch = data[offset:offset + batch]
            att_batch = att_data[offset:offset + batch]
            h_batch = h[:batch]
            h_new = self.cell(x_batch, h_batch, att_batch)
            outputs.append(h_new)
            h = h_new
            offset += batch
        out = torch.cat(outputs, dim=0)
        return PackedSequence(out, batch_sizes, sorted_idx, unsorted_idx)