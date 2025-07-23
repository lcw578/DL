import torch
from torchvision import transforms,datasets
from torch.utils.data.dataloader import DataLoader
from net import mixed_net

def test_model(model_path, test_loader, dataset_name):
    """
    测试模型性能
    Args:
        model_path: 模型文件路径
        test_loader: 测试数据加载器
        dataset_name: 数据集名称
    """
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载模型
    model = mixed_net()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  

    # 初始化计数器
    class_correct = list(0. for i in range(3))
    class_total = list(0. for i in range(3))
    class_names = ['blue', 'red', 'yellow']  # 根据你的数据集类别

    # 开始测试
    with torch.no_grad():
        for batch_idx, (datas, labels) in enumerate(test_loader):
            datas, labels = datas.to(device), labels.to(device)
            
            outputs = model(datas)
            _, predicted = torch.max(outputs, 1)

            # 统计每个类别的正确率
            matches = (predicted == labels)
            for i in range(len(labels)):
                label = labels[i].item()
                class_correct[label] += matches[i].item()
                class_total[label] += 1

    # 打印结果
    print(f"\n=== {dataset_name} 测试结果 ===")
    
    # 计算总体正确率
    total_correct = sum(class_correct)
    total = sum(class_total)
    overall_accuracy = 100.0 * total_correct / total
    print(f'总体正确率: {overall_accuracy:.2f}%')

    # 输出每个类别的正确率
    for i in range(3):
        if class_total[i] > 0:
            class_accuracy = 100 * class_correct[i] / class_total[i]
            print(f'{class_names[i]} 类别正确率: {class_accuracy:.2f}% ({int(class_correct[i])}/{int(class_total[i])})')
        else:
            print(f'{class_names[i]} 类别: 无测试样本')
    
    return overall_accuracy

if __name__ == "__main__":
    # 图像转换 (与训练时保持一致)
    transform = transforms.Compose([
        transforms.Resize([64, 64]),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 加载测试数据
    BATCH_SIZE = 512  # 测试时可以用较大batch size
    testset1 = datasets.ImageFolder(root=r'dataset/test1', transform=transform)
    testset2 = datasets.ImageFolder(root=r'dataset/test2', transform=transform)
    
    test_loader1 = DataLoader(testset1, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
    test_loader2 = DataLoader(testset2, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    print(f"测试集1样本数: {len(testset1)}")
    print(f"测试集2样本数: {len(testset2)}")
    print(f"类别映射: {testset1.class_to_idx}")

    # 测试最佳模型
    model_path = r"pth/model2_balanced_avg96.5_min95.7_e170.pth"
    
    try:
        acc1 = test_model(model_path, test_loader1, "Test Set 1")
        acc2 = test_model(model_path, test_loader2, "Test Set 2")
        
        print(f"\n=== 综合测试结果 ===")
        print(f"测试集1正确率: {acc1:.2f}%")
        print(f"测试集2正确率: {acc2:.2f}%")
        print(f"平均正确率: {(acc1 + acc2) / 2:.2f}%")
        
    except FileNotFoundError:
        print(f"模型文件未找到: {model_path}")
        print("请先训练模型或检查文件路径")