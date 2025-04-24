import csv
import random
from collections import defaultdict
from typing import Dict
from torch.utils.data import Dataset

class UserItemInteractionDataset(Dataset):
    """
    用户-商品交互数据集(无标题CSV版本)
    列顺序: 用户ID,商品ID,商品类目ID,行为类型,时间戳
    """
    
    def __init__(self, csv_path: str):
        # 存储用户最后一次交互(完整记录)
        self.user_last_interactions = {}
        # 存储所有商品记录(用于负采样)
        self.all_item_records = []
        # 按商品ID索引的记录
        self.item_records_dict = defaultdict(list)
        
        # 用于计算最大值的临时变量
        max_item_id = 0
        max_category_id = 0
        
        # 第一次遍历：读取原始数据并计算最大值
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                item_id = float(row[1])
                category_id = float(row[2])
                if item_id > max_item_id:
                    max_item_id = item_id
                if category_id > max_category_id:
                    max_category_id = category_id
        
        # 第二次遍历：进行归一化处理
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                record = {
                    'user_id': int(row[0]),
                    'item_id': float(row[1]) / max_item_id,  # 使用item_id的最大值归一化
                    'category_id': float(row[2]) / max_category_id,  # 使用category_id的最大值归一化
                    'behavior_type': row[3],
                    'timestamp': int(row[4])
                }
                
                # 存储所有商品记录(用于负采样)
                self.all_item_records.append(record)
                self.item_records_dict[record['item_id']].append(record)
                
                # 更新用户最后一次交互
                user_id = record['user_id']
                if (user_id not in self.user_last_interactions or 
                    record['timestamp'] > self.user_last_interactions[user_id]['timestamp']):
                    self.user_last_interactions[user_id] = record
        
        # 转换为用户ID列表便于索引
        self.user_ids = list(self.user_last_interactions.keys())
    
    def __len__(self) -> int:
        return len(self.user_ids)
    
    def __getitem__(self, idx: int) -> Dict:
        user_id = self.user_ids[idx]
        pos_sample = self.user_last_interactions[user_id]
        
        # 从全体商品中随机选择一个负样本
        neg_record = random.choice(self.all_item_records)
        
        # 构建负样本(保持相同时间戳)
        neg_sample = {
            'item_id': neg_record['item_id'],
            'category_id': neg_record['category_id'],
            'behavior_type': 'negative',
            'timestamp': pos_sample['timestamp']
        }

        return {
            'user_id': user_id,
            'positive_sample': pos_sample,
            'negative_sample': neg_sample
        }