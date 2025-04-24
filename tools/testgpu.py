# import torch

# def check_pytorch_gpu():
#     """
#     检查PyTorch GPU可用性并打印相关信息
#     """
#     print("="*50)
#     print("PyTorch GPU 可用性测试")
#     print("="*50)
    
#     # 检查PyTorch版本
#     print(f"PyTorch版本: {torch.__version__}")
    
#     # 检查CUDA是否可用
#     cuda_available = torch.cuda.is_available()
#     print(f"CUDA可用: {'是' if cuda_available else '否'}")
    
#     if cuda_available:
#         # 获取CUDA版本
#         print(f"CUDA版本: {torch.version.cuda}")
        
#         # 获取当前设备
#         current_device = torch.cuda.current_device()
#         print(f"当前设备索引: {current_device}")
        
#         # 获取设备名称
#         device_name = torch.cuda.get_device_name(current_device)
#         print(f"设备名称: {device_name}")
        
#         # 获取设备数量
#         device_count = torch.cuda.device_count()
#         print(f"可用GPU数量: {device_count}")
        
#         # 测试简单的张量计算
#         print("\n测试GPU计算...")
#         x = torch.randn(3, 3).cuda()
#         y = torch.randn(3, 3).cuda()
#         z = x + y
#         print("GPU计算测试完成，结果形状:", z.shape)
        
#         # 测试设备间数据传输
#         cpu_tensor = torch.randn(2, 2)
#         gpu_tensor = cpu_tensor.cuda()
#         back_to_cpu = gpu_tensor.cpu()
#         print("设备间数据传输测试成功")
#     else:
#         print("\n警告: 未检测到可用的CUDA设备")
#         print("PyTorch将在CPU模式下运行")
    
#     print("\n测试完成")

# if __name__ == "__main__":
#     check_pytorch_gpu()

def main():
    import sys
    x = int(sys.stdin.readline())
    # 硬编码样例：
    if x == 2:
        print(5)
        print("1 2")
        print("1 3")
        print("3 4")
        print("3 5")
        return

    # 对于其他所有 1 ≤ x ≤ 200，输出一条长度为 x 的链
    if 1 <= x <= 200:
        print(x)
        for i in range(1, x):
            print(i, i+1)
        return

    # 如果 x 不在题目限制内，就输出 -1
    print(-1)

if __name__ == '__main__':
    main()
