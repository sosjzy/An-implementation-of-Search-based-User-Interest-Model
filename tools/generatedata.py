import pandas as pd

input_csv  = '/openbayes/home/data/UserBehavior.csv'
output_csv = 'output_sampled.csv'
n_users    = 300000

# 1. 不带表头读取，指定列名
df = pd.read_csv(input_csv, header=None, names=['user_id', 'col2', 'col3', 'col4', 'col5'])

# 2. 随机抽取 n_users 个唯一 user_id
#    random_state 保证可复现
sampled_ids = df['user_id'].drop_duplicates().sample(n=n_users, random_state=42)

# 3. 筛出这些 user_id 的所有行
out_df = df[df['user_id'].isin(sampled_ids)]

# 4. 写入新文件（不写入行索引）
out_df.to_csv(output_csv, header=False, index=False)
