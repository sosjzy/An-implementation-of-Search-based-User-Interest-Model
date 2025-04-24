import pickle
from collections import defaultdict
import csv

# # 读取CSV并构建用户ID到交互记录的字典
# user_interactions = defaultdict(list)

# with open("/openbayes/home/code/sim/filtered_interactions200.csv", "r") as f:
#     reader = csv.reader(f)
#     for row in reader:
#         user_id = row[0]
#         user_interactions[user_id].append(row)  # 将记录按用户ID分组

# 保存为PKL文件
# with open("user_interactions.pkl", "wb") as f:
#     pickle.dump(user_interactions, f)

# with open("/openbayes/home/data/user_interactions.pkl", "rb") as f:
#     data = pickle.load(f)

# # 查找用户ID="123"的所有交互记录
# user_id = "1000633"
# interactions = data.get(user_id, [])
# print(f"用户 {user_id} 有 {len(interactions)} 条交互记录")