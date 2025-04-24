from layers.sequence import AUGRUCell
import torch

cell = AUGRUCell(input_size=16, hidden_size=16)
x = torch.randn(4, 16)
h0 = torch.zeros(4, 16)
att = torch.rand(4)
h1 = cell(x, h0, att)
print(h1.shape)  # 应该是 (4,16)
