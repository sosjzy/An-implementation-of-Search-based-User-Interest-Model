#!/usr/bin/env python3
"""
test_key_consistency.py

检查 DataLoader 中取到的 user_id 是否能在 user_interactions.pkl 中找到对应序列，
以及检验使用 str(uid) 或 uid 作为键时的差异。
"""

import pickle
import sys
from database import UserItemInteractionDataset

# 配置：请根据实际路径修改
PICKLE_FILE = '/openbayes/home/data/user_interactions.pkl'
CSV_FILE    = '/openbayes/home/data/latest_interactions.csv'
NUM_CHECK   = 10  # 检查前 NUM_CHECK 个样本

def main():
    # 1) 加载 pickle
    try:
        with open(PICKLE_FILE, 'rb') as f:
            user_interactions = pickle.load(f)
    except Exception as e:
        print(f"Error loading pickle '{PICKLE_FILE}': {e}", file=sys.stderr)
        sys.exit(1)
    print(f"Loaded pickle with {len(user_interactions)} users")

    # 2) 加载数据集
    ds = UserItemInteractionDataset(CSV_FILE)

    # 3) 检查前 NUM_CHECK 条记录的 user_id
    print(f"\nChecking first {NUM_CHECK} dataset entries:")
    for idx in range(min(NUM_CHECK, len(ds))):
        sample = ds[idx]

        uid_tensor = sample['user_id']
        # 如果 user_id 是 Tensor，则取 .item()
        uid = uid_tensor.item() if hasattr(uid_tensor, 'item') else uid_tensor
        print(f"\nDataset idx={idx}: user_id (type={type(uid)}) = {uid}")

        # 尝试用 int 和 str 两种方式在 pickle 中查找
        for key in (uid, str(uid)):
            exists = key in user_interactions
            print(user_interactions[key])
            length = len(user_interactions[key]) if exists else None
            print(f"  key = {repr(key):>12}  found: {exists:5}  seq_length: {length}")

    print("\nIf 'found' is False for both int and str, then this user_id is missing in the pickle.")
    print("If only int or only str is True, adjust your lookup to use that key type.")

if __name__ == '__main__':
    main()
