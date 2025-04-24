import os
import time
import pickle
import random
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from database import UserItemInteractionDataset
from model import SimpleRecommendationModel



def print_config(cfg):
    print("=" * 80)
    for k, v in cfg.items():
        print(f"{k:15s}: {v}")
    print("=" * 80)


def build_histories(batch, userdata, device, short_K=100, long_K=100):
    """
    构建短期和长期历史序列：仅保留 ts < 当前样本时间
    返回：
      shorttime: Tensor[B, H, 2]
      st_lens:   Tensor[B]
      longtime:  Tensor[B, L, 3]
      lt_lens:   Tensor[B]
    """
    user_ids = batch['user_id'].tolist()
    pos = batch['positive_sample']
    pos_cats  = pos['category_id'].tolist()
    pos_times = pos['timestamp'].tolist()

    st_seqs, st_lens = [], []
    lt_seqs, lt_lens = [], []

    for uid, pc, pts in zip(user_ids, pos_cats, pos_times):
        hist = userdata[int(uid)]
        hist_before = [r for r in hist if int(r[4]) < pts]

        if len(hist_before) >= short_K:
            st_window = hist_before[-short_K:]
            early = hist_before[:-short_K]
            lt = [
                [int(r[1]), int(r[2]), pts - int(r[4])]
                for r in early if int(r[2]) == pc
            ]
            lt = lt[-long_K:] or [[0,0,0]]
        else:
            st_window = hist_before
            lt = []

        st = [[int(r[1]), int(r[2])] for r in st_window] or [[0,0]]
        st_seqs.append(st); st_lens.append(len(st))
        lt_seqs.append(lt); lt_lens.append(len(lt))

    # pad 短期
    H = max(st_lens)
    st_padded = [s + [[0,0]]*(H-len(s)) for s in st_seqs]
    shorttime = torch.tensor(st_padded, dtype=torch.long, device=device)
    st_lens = torch.tensor(st_lens, dtype=torch.long, device=device).clamp(min=1)

    # pad 长期
    if any(lt_lens):
        L = max(lt_lens)
        lt_padded = [s + [[0,0,0]]*(L-len(s)) for s in lt_seqs]
        longtime = torch.tensor(lt_padded, dtype=torch.long, device=device)
        lt_lens = torch.tensor(lt_lens, dtype=torch.long, device=device).clamp(min=1)
    else:
        B = len(user_ids)
        longtime = torch.zeros((B,1,3), dtype=torch.long, device=device)
        lt_lens = torch.zeros(B, dtype=torch.long, device=device)

    return shorttime, st_lens, longtime, lt_lens


def test_once_realistic(
    test_csv: str,
    userdata: dict,
    model: nn.Module,
    device: torch.device,
    neg_k: int = 1,
    batch_size: int = 64
) -> Tuple[float, float]:
    # 1) 读取所有正样本，构建全量 item 列表 & item->category 映射
    pos_df = pd.read_csv(
        test_csv,
        header=None,
        names=['user_id','item_id','category_id','action','timestamp']
    )
    all_items = pos_df['item_id'].unique().tolist()
    item2cat = dict(zip(pos_df['item_id'], pos_df['category_id']))

    # 2) 保持原 Dataset 只产正样本
    ds = UserItemInteractionDataset(test_csv)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    model.eval()

    total_loss = 0.0
    all_labels = []
    all_probs  = []

    with torch.no_grad():
        for batch in loader:
            user_ids  = batch['user_id'].tolist()
            pos       = batch['positive_sample']
            pos_items = pos['item_id'].tolist()
            pos_cats  = pos['category_id'].tolist()
            pos_times = pos['timestamp'].tolist()
            B = len(user_ids)

            # 负样本采样
            neg_items_batch = []
            neg_cats_batch  = []
            for uid, pc in zip(user_ids, pos_cats):
                hist_items = {int(r[1]) for r in userdata[int(uid)]}
                candidates = list(set(all_items) - hist_items)
                sampled = random.sample(candidates, neg_k)
                neg_items_batch.append(sampled)
                neg_cats_batch.append([item2cat[i] for i in sampled])

            # 构造 feature tensors
            pos_feat = torch.tensor(
                np.stack([pos_items, pos_cats], axis=1),
                dtype=torch.float, device=device
            )
            neg_feat = torch.tensor(
                np.stack([sum(neg_items_batch, []), sum(neg_cats_batch, [])], axis=1),
                dtype=torch.float, device=device
            )

            # 构造历史序列
            shorttime, st_lens, longtime, lt_lens = build_histories(batch, userdata, device)

            # forward + loss
            pos_logits, neg_logits = model(
                pos_feat, neg_feat,
                shorttime, st_lens,
                longtime, lt_lens
            )
            logits = torch.cat([pos_logits, neg_logits], dim=0)

            # 正样本 label=1，负样本 label=0
            labels = torch.cat([
                torch.ones(B, dtype=torch.long, device=device),
                torch.zeros(B*neg_k, dtype=torch.long, device=device)
            ], dim=0)

            loss = criterion(logits, labels)
            total_loss += loss.item() * (B*(1+neg_k))

            probs = torch.softmax(logits, dim=1)[:,1].cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_probs  = np.concatenate(all_probs)
    avg_loss = total_loss / (len(ds) * (1 + neg_k))
    auc      = roc_auc_score(all_labels, all_probs)

    return avg_loss, auc


class Trainer:
    """带最近(shorttime)和更早(longtime)历史的训练器"""

    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 userdata: dict,
                 device: torch.device,
                 lr: float):
        self.device = device
        self.model = model.to(device)
        self.train_loader = train_loader
        self.userdata = userdata
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train_epoch(self, epoch_idx: int) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        all_probs, all_labels = [], []

        pbar = tqdm(self.train_loader, desc=f"[Epoch {epoch_idx}]", ncols=100)
        for batch in pbar:
            start = time.time()

            user_ids = batch['user_id'].tolist()
            pos = batch['positive_sample']
            neg = batch['negative_sample']
            pos_cats  = pos['category_id'].tolist()
            pos_times = pos['timestamp'].tolist()

            st_seqs, st_lens = [], []
            lt_seqs, lt_lens = [], []

            for uid, pc, pts in zip(user_ids, pos_cats, pos_times):
                hist = self.userdata[int(uid)]
                hist_before = [r for r in hist if int(r[4]) < pts]

                if len(hist_before) >= 100:
                    st_window = hist_before[-100:]
                    early = hist_before[:-100]
                    lt = [
                        [int(r[1]), int(r[2]), pts - int(r[4])] for r in early
                        if int(r[2]) == pc
                    ]
                    lt = lt[-100:] or [[0,0,0]]
                else:
                    st_window = hist_before
                    lt = []

                st = [[int(r[1]), int(r[2])] for r in st_window] or [[0,0]]
                st_seqs.append(st); st_lens.append(len(st))
                lt_seqs.append(lt); lt_lens.append(len(lt))

            H = max(st_lens)
            st_padded = [s + [[0,0]]*(H-len(s)) for s in st_seqs]
            shorttime = torch.tensor(st_padded, dtype=torch.long, device=self.device)
            st_lens = torch.tensor(st_lens, dtype=torch.long, device=self.device).clamp(min=1)

            L = max(lt_lens)
            lt_padded = [s + [[0,0,0]]*(L-len(s)) for s in lt_seqs]
            longtime = torch.tensor(lt_padded, dtype=torch.long, device=self.device)
            lt_lens = torch.tensor(lt_lens, dtype=torch.long, device=self.device).clamp(min=1)

            pos_feat = torch.stack([pos['item_id'], pos['category_id']], dim=1).float().to(self.device)
            neg_feat = torch.stack([neg['item_id'], neg['category_id']], dim=1).float().to(self.device)
            B = pos_feat.size(0)

            pos_logits, neg_logits = self.model(
                pos_feat, neg_feat,
                shorttime, st_lens,
                longtime, lt_lens
            )
            logits = torch.cat([pos_logits, neg_logits], dim=0)
            labels = torch.cat([
                torch.zeros(B, dtype=torch.long, device=self.device),
                torch.ones(B, dtype=torch.long, device=self.device)
            ], dim=0)

            loss = self.criterion(logits, labels)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()

            total_loss += loss.item() * (2 * B)

            probs = torch.softmax(logits, dim=1)[:,1].detach().cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.cpu().numpy())

            pos_p = torch.softmax(pos_logits, dim=1)[:,0].mean().item()
            neg_p = torch.softmax(neg_logits, dim=1)[:,1].mean().item()
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'pos_p': f"{pos_p:.4f}",
                'neg_p': f"{neg_p:.4f}",
                't':     f"{time.time()-start:.2f}s"
            })

        epoch_loss = total_loss / (len(self.train_loader.dataset) * 2)
        epoch_auc  = roc_auc_score(
            np.concatenate(all_labels),
            np.concatenate(all_probs)
        )
        return epoch_loss, epoch_auc


def main():
    cfg = {
        'data_path':       '/openbayes/home/data/train_split.csv',
        'userdata_path':   '/openbayes/home/data/user_interactions.pkl',
        'test_csv':        '/openbayes/home/data/test_split.csv',
        'batch_size':      1024,
        'num_epochs':      5,
        'lr':              1e-3,
        'decay_gamma':     0.5,
        'embedding_dim':   4,
        'device':          torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
        'model_save_path': 'best_model.pth',
    }

    print_config(cfg)

    with open(cfg['userdata_path'], 'rb') as f:
        userdata = pickle.load(f)

    dataset = UserItemInteractionDataset(cfg['data_path'])
    loader  = DataLoader(
        dataset,
        batch_size=cfg['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=(cfg['device'].type=='cuda')
    )

    model   = SimpleRecommendationModel(embedding_dim=cfg['embedding_dim'])
    trainer = Trainer(model, loader,
                      userdata,
                      cfg['device'],
                      cfg['lr'])
    scheduler = ExponentialLR(trainer.optimizer, gamma=cfg['decay_gamma'])

    best_loss = float('inf')
    for epoch in range(1, cfg['num_epochs']+1):
        start = time.time()
        train_loss, train_auc = trainer.train_epoch(epoch)
        scheduler.step()
        elapsed = time.time() - start
        lr = trainer.optimizer.param_groups[0]['lr']

        print(f"\nEpoch {epoch:2d} | lr={lr:.5f} | "
              f"train_loss={train_loss:.4f} | train_auc={train_auc:.4f} | "
              f"time={elapsed:.2f}s")

        test_loss, test_auc = test_once_realistic(
            cfg['test_csv'],
            userdata,
            model,
            cfg['device'],
            neg_k=1,
            batch_size=cfg['batch_size']
        )
        print(f"Epoch {epoch:2d} | test_loss={test_loss:.4f} | test_auc={test_auc:.4f}\n")

        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(model.state_dict(), cfg['model_save_path'])
            print("⇒ Saved best model\n")

    print(f"Training done! Best loss = {best_loss:.4f}")


if __name__ == '__main__':
    main()
