import torch
import torch.nn as nn

# 定义一个测试函数来验证 nn.CrossEntropyLoss（针对二分类）
def test_binary_cross_entropy_loss():
    # 假设 batch_size = 4
    batch_size = 4
    output_dim = 2  # 二分类，输出两个类别的概率（logits）

    # 随机生成模型的输出 logits，形状为 (batch_size, output_dim)
    # logits 是未经过 softmax 归一化的原始分数
    outputs = torch.randn(batch_size, output_dim)

    # 假设标签全为 1（表示所有样本为正类）
    labels = torch.zeros(batch_size, dtype=torch.long)

    # 创建 CrossEntropyLoss 函数
    criterion = nn.CrossEntropyLoss()

    # 计算交叉熵损失
    loss = criterion(outputs, labels)

    # 打印结果
    print(f"模型输出 logits: \n{outputs}")
    print(f"标签: \n{labels}")
    print(f"交叉熵损失: {loss.item()}")

    return loss

# 调用测试函数
loss = test_binary_cross_entropy_loss()
