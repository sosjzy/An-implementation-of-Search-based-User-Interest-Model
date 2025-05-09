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

        # 投影层：1 -> D，用 Xavier 初始化
        self.input_proj = nn.Embedding(5200000, embedding_dim)
        nn.init.xavier_uniform_(self.input_proj.weight)

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
        # 确保短期和长期数据是三维的，并且计算 lengths
        if shorttime is not None and shorttime.dim() == 2:
            # 如果 shorttime 是 2D，则扩展为 3D
            shorttime = shorttime.unsqueeze(0)
            lengths = torch.tensor([shorttime.size(1)], device=shorttime.device)  # 获取时间步数

        if longtime is not None and longtime.dim() == 2:
            # 如果 longtime 是 2D，则扩展为 3D
            longtime = longtime.unsqueeze(0)
            lt_lengths = torch.tensor([longtime.size(1)], device=longtime.device)  # 获取时间步数

        # 短期序列 embedding: (B, Hs, 2) -> (B, Hs, D)
        
        st = shorttime.long()  # 保证 shorttime 是 LongTensor
        # print(st.shape)
        st_last_dim = st[..., -1]  # 获取最后一维，商品类别编号
        B, Hs, _ = st.size()
        st_emb = self.input_proj(st_last_dim.view(-1, 1)).view(B, Hs, -1)

        if longtime is not None:
            lt = longtime.long()  # 保证 longtime 是 LongTensor

            # 检查 lt 的维度
            if lt.dim() != 3 or lt.size(2) != 3:
                print("none")  # 如果维度不对，输出 'none'
                lt_emb = None  # 设置 lt_emb 为 None
            else:
                _, Hl, _ = lt.size()
                # 拆分 idcat 与 time_diff
                idcat = lt[:, :, :2].contiguous().view(-1, 2)  # (B * Hl, 2)
                lt_last_dim = idcat[..., -1].view(-1, 1)  # 获取最后一维，商品类别编号
                time_diff = lt[:, :, 2].unsqueeze(-1).contiguous().view(-1, 1)  # (B * Hl, 1)
                # 投影并融合
                idcat_emb = self.input_proj(lt_last_dim).view(B, Hl, -1)  # (B, Hl, D)
                time_emb = self.time_proj(time_diff.float()).view(B, Hl, -1)  # 转换为 float 类型
                lt_emb = idcat_emb + time_emb  # (B, Hl, D)
        else:
            lt_emb = None

        # 正负样本 embedding
        pi = positive_items[..., -1].view(-1, 1).long()  # (B, 1) 转为 LongTensor
        pos_emb = self.input_proj(pi)  # (B, D)
        pos_emb = pos_emb.squeeze(1)  # 去除多余的维度
        
        ni = negative_items[..., -1].view(-1, 1).long()  # (B, 1) 转为 LongTensor
        neg_emb = self.input_proj(ni)  # (B, D)
        neg_emb = neg_emb.squeeze(1)  # 去除多余的维度

        # 短期兴趣
        short_interest = self.st_extractor(st_emb, pos_emb, lengths)  # (B, D)
        st_neg = self.st_extractor(st_emb, neg_emb, lengths)  # (B, D)

        # 长期兴趣 via ESU
        if lt_emb is not None:
            long_interest = self.esu(pos_emb, lt_emb, lt_lengths)  # (B, D)
            lt_neg = self.esu(neg_emb, lt_emb, lt_lengths)  # (B, D)
        else:
            long_interest = torch.zeros_like(short_interest)  # 如果没有长期兴趣，则全零
            lt_neg = torch.zeros_like(short_interest)  # 如果没有长期兴趣，则全零

        # 融合并输出 logits
        def compute_logits(item_emb, short_interest, long_interest):
            feat = torch.cat([item_emb, short_interest, long_interest], dim=1)  # (B, 3D)
            return self.fcn(feat)  # (B, 2)

        pos_logits = compute_logits(pos_emb, short_interest, long_interest)
        neg_logits = compute_logits(neg_emb, st_neg, lt_neg)
        return pos_logits, neg_logits
