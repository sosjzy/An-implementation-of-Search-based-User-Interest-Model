import pandas as pd
import pickle
import random
from collections import defaultdict

# 配置：请根据实际路径修改
INPUT_CSV        = '/openbayes/home/data/output_sampled.csv'
PICKLE_OUT       = '/openbayes/home/data/user_interactions.pkl'
RANDOM_CSV_OUT   = '/openbayes/home/data/random_interactions_1M.csv'
CHUNK_SIZE       = 5000000     # 每次读取行数
MIN_INTERACT     = 0             # 最少交互次数阈值
SAMPLE_SIZE      = 1000000     # 随机抽取行数

def build_user_sequences():
    # —— 第 1 遍扫描：统计每个用户的交互次数 —— #
    user_counts = {}
    for chunk in pd.read_csv(
        INPUT_CSV,
        header=None,
        names=['user_id','item_id','category_id','action','timestamp'],
        usecols=[0,1,2,3,4],
        chunksize=CHUNK_SIZE
    ):
        vc = chunk['user_id'].value_counts()
        for uid, cnt in vc.items():
            user_counts[uid] = user_counts.get(uid, 0) + int(cnt)

    # 筛出交互次数 > MIN_INTERACT 的用户集合
    valid_users = {uid for uid, cnt in user_counts.items() if cnt > MIN_INTERACT}
    print(f"用户总数: {len(user_counts)}, 交互次数 > {MIN_INTERACT} 的用户: {len(valid_users)}")

    # —— 第 2 遍扫描：收集每个有效用户的完整交互序列 & 水塘抽样 —— #
    user_interactions = defaultdict(list)
    reservoir = []
    total_kept = 0

    for chunk in pd.read_csv(
        INPUT_CSV,
        header=None,
        names=['user_id','item_id','category_id','action','timestamp'],
        usecols=[0,1,2,3,4],
        chunksize=CHUNK_SIZE
    ):
        sub = chunk[chunk['user_id'].isin(valid_users)]
        for row in sub.itertuples(index=False):
            # 1) 完整序列
            rec = [row.user_id, row.item_id, row.category_id, row.action, row.timestamp]
            user_interactions[row.user_id].append(rec)

            # 2) 水塘抽样
            total_kept += 1
            if len(reservoir) < SAMPLE_SIZE:
                reservoir.append(rec)
            else:
                j = random.randrange(total_kept)
                if j < SAMPLE_SIZE:
                    reservoir[j] = rec

    # —— 保存所有有效用户的交互序列到 pickle —— #
    with open(PICKLE_OUT, 'wb') as f:
        pickle.dump(dict(user_interactions), f)
    print(f"已保存用户交互序列到: {PICKLE_OUT}")

    # —— 将随机抽出的 1,000,000 行写入 CSV —— #
    import csv
    with open(RANDOM_CSV_OUT, 'w', newline='', encoding='utf-8') as fout:
        writer = csv.writer(fout)
        # 没有表头
        writer.writerows(reservoir)
    print(f"已保存随机抽样 {SAMPLE_SIZE} 行到: {RANDOM_CSV_OUT}")

if __name__ == '__main__':
    build_user_sequences()
