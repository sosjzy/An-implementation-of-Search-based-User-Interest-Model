import torch
import torch.nn as nn
from typing import Tuple, Optional
from dien import ShortTermInterestExtractor
from ESU import ESU  # 引入多头交叉注意力模块

class SimpleRecommendationModel(nn.Module):
    """
    简单推荐模型，融合短期(shorttime) & 长期(longtime)交互兴趣

    输入:
        positive_items: Tensor (B, 2)
        negative_items: Tensor (B, 2)
        shorttime:      Tensor (B, Hs, 2)
        lengths:        Tensor (B,) 或 None
        longtime:       Tensor (B, Hl, 3)  (item_id, category_id, time_diff)
        lt_lengths:     Tensor (B,) 或 None
    输出:
        pos_logits: Tensor (B, 2)
        neg_logits: Tensor (B, 2)
    """

    def __init__(self, embedding_dim: int = 16, num_heads: int = 4):
        super().__init__()
        self.embedding_dim = embedding_dim

        # 投影层：2 -> D，用 Xavier 初始化
        self.input_proj = nn.Linear(2, embedding_dim)
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)

        # 时间差投影：1 -> D
        self.time_proj = nn.Linear(1, embedding_dim)
        nn.init.xavier_uniform_(self.time_proj.weight)
        nn.init.zeros_(self.time_proj.bias)

        # 短期兴趣提取器
        self.st_extractor = ShortTermInterestExtractor(
            input_size=embedding_dim, hidden_size=embedding_dim,
            num_layers=1, bidirectional=False
        )
        # 长期兴趣提取器改为 ESU
        self.esu = ESU(embed_dim=embedding_dim, num_heads=num_heads)

        # FFN：融合 item_emb + short_interest + long_interest 再映射到 logits
        self.fcn = nn.Sequential(
            nn.Linear(3 * embedding_dim, 200),
            nn.PReLU(),
            nn.Linear(200, 80),
            nn.PReLU(),
            nn.Linear(80, 2),
        )
        for m in self.fcn:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        positive_items: torch.Tensor,
        negative_items: torch.Tensor,
        shorttime: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        longtime: Optional[torch.Tensor] = None,
        lt_lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 支持单样本时自动扩维
        if shorttime is not None and shorttime.dim() == 2:
            shorttime = shorttime.unsqueeze(0)
            lengths = torch.tensor([shorttime.size(1)], device=shorttime.device)
        if longtime is not None and longtime.dim() == 2:
            longtime = longtime.unsqueeze(0)
            lt_lengths = torch.tensor([longtime.size(1)], device=longtime.device)

        # 短期序列 embedding: (B, Hs, 2)->(B, Hs, D)
        st = shorttime.float()
        B, Hs, _ = st.size()
        st_emb = self.input_proj(st.view(-1, 2)).view(B, Hs, -1)

        # 长期序列 embedding: (B, Hl, 3)
        if longtime is not None:
            lt = longtime.float()
            _, Hl, _ = lt.size()
            # 拆分 idcat 与 time_diff
            idcat = lt[:, :, :2].contiguous().view(-1, 2)       # (B*Hl,2)
            time_diff = lt[:, :, 2].unsqueeze(-1).contiguous().view(-1, 1)  # (B*Hl,1)
            # 投影并融合
            idcat_emb = self.input_proj(idcat).view(B, Hl, -1)
            time_emb  = self.time_proj(time_diff).view(B, Hl, -1)
            lt_emb = idcat_emb + time_emb  # (B, Hl, D)
        else:
            lt_emb = None

        # 正负样本 embedding
        pos_emb = self.input_proj(positive_items.float())  # (B, D)
        neg_emb = self.input_proj(negative_items.float())  # (B, D)

        # 短期兴趣
        short_interest = self.st_extractor(st_emb, pos_emb, lengths)  # (B, D)
        # 长期兴趣 via ESU
        if lt_emb is not None:
            long_interest = self.esu(pos_emb, lt_emb, lt_lengths)  # (B, D)
        else:
            long_interest = torch.zeros_like(short_interest)

        # 融合并输出 logits
        def compute_logits(item_emb):
            feat = torch.cat([item_emb, short_interest, long_interest], dim=1)  # (B,3D)
            return self.fcn(feat)  # (B,2)

        pos_logits = compute_logits(pos_emb)
        neg_logits = compute_logits(neg_emb)
        return pos_logits, neg_logits
