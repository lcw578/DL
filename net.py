import torch
from torch import nn
from torchvision import transforms,datasets
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from torchinfo import summary
import os

class mixed_net(nn.Module):
    def __init__(self):
        super(mixed_net,self).__init__()
        # 第一个卷积块 - 输入: (batch_size, 3, 64, 64)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # 输入3通道(RGB)，输出32通道 -> (batch_size, 32, 64, 64)
        self.bn1 = nn.BatchNorm2d(32)  # 批归一化，维度不变 -> (batch_size, 32, 64, 64)
        self.pool1 = nn.MaxPool2d(2, 2)  # 最大池化，尺寸减半 -> (batch_size, 32, 32, 32)
        
        # 第二个卷积块 - 输入: (batch_size, 32, 32, 32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 通道数32->64，尺寸不变 -> (batch_size, 64, 32, 32)
        self.bn2 = nn.BatchNorm2d(64)  # 批归一化，维度不变 -> (batch_size, 64, 32, 32)
        self.pool2 = nn.MaxPool2d(2, 2)  # 最大池化，尺寸减半 -> (batch_size, 64, 16, 16)
        
        # 第三个卷积块 - 输入: (batch_size, 64, 16, 16)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # 通道数64->128，尺寸不变 -> (batch_size, 128, 16, 16)
        self.bn3 = nn.BatchNorm2d(128)  # 批归一化，维度不变 -> (batch_size, 128, 16, 16)
        self.pool3 = nn.MaxPool2d(2, 2)  # 最大池化，尺寸减半 -> (batch_size, 128, 8, 8)
        
        # 全连接层 - 输入: (batch_size, 128*8*8=8192)
        self.fc1 = nn.Linear(128 * 8 * 8, 512)  # 全连接层8192->512 -> (batch_size, 512)
        self.dropout1 = nn.Dropout(0.5)  # Dropout，维度不变 -> (batch_size, 512)
        # 暂退法在前向传播过程中，计算每一内部层的同时丢弃一些神经元。
        # 暂退法可以避免过拟合，它通常与控制权重向量的维数和大小结合使用的。
        self.fc2 = nn.Linear(512, 128)  # 全连接层512->128 -> (batch_size, 128)
        self.dropout2 = nn.Dropout(0.3)  # Dropout，维度不变 -> (batch_size, 128)
        self.fc3 = nn.Linear(128, 3)  # 输出层128->3个类别 -> (batch_size, 3)

    
    def forward(self, x):
        '''
        前向传播过程，详细尺寸变化：
        公式： W = (W + 2*padding - kernel_size) / stride + 1
        输入: (batch_size, 3, 64, 64)
        '''
        # 第一个卷积块
        # Conv1: (batch_size, 3, 64, 64) -> (batch_size, 32, 64, 64)
        # BN1 + ReLU: (batch_size, 32, 64, 64) -> (batch_size, 32, 64, 64)
        # Pool1: (batch_size, 32, 64, 64) -> (batch_size, 32, 32, 32)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        
        # 第二个卷积块
        # Conv2: (batch_size, 32, 32, 32) -> (batch_size, 64, 32, 32)
        # BN2 + ReLU: (batch_size, 64, 32, 32) -> (batch_size, 64, 32, 32)
        # Pool2: (batch_size, 64, 32, 32) -> (batch_size, 64, 16, 16)
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        
        # 第三个卷积块
        # Conv3: (batch_size, 64, 16, 16) -> (batch_size, 128, 16, 16)
        # BN3 + ReLU: (batch_size, 128, 16, 16) -> (batch_size, 128, 16, 16)
        # Pool3: (batch_size, 128, 16, 16) -> (batch_size, 128, 8, 8)
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        # 展平操作
        # Flatten: (batch_size, 128, 8, 8) -> (batch_size, 128*8*8=8192)
        x = x.view(x.size(0), -1)
        
        # 全连接层
        # FC1: (batch_size, 8192) -> (batch_size, 512)
        # ReLU + Dropout1: (batch_size, 512) -> (batch_size, 512)
        x = F.relu(self.fc1(x))#使用relu作为激活函数
        x = self.dropout1(x)#dropout减少过拟合
        
        # FC2: (batch_size, 512) -> (batch_size, 128)
        # ReLU + Dropout2: (batch_size, 128) -> (batch_size, 128)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        
        # 输出层
        # FC3: (batch_size, 128) -> (batch_size, 3) [3个类别的logits]
        x = self.fc3(x)
        
        # Softmax回归 - 将logits转换为概率分布
        x = F.softmax(x, dim=1)

        return x  # 最终输出: (batch_size, 3) [3个类别的概率]

if __name__ == "__main__":
    #图像转换
    transforms = transforms.Compose(
        [
            transforms.Resize([64, 64]),#统一输入尺寸
            transforms.ToTensor(),#将图片转换为Tensor
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))#对每个通道进行归一化处理
        ]
    )
    
    #超参数设置
    BATCH_SIZE = 1024#批处理大小
    EPOCH = 200#训练轮数

    #加载数据
    trainset = datasets.ImageFolder(root=r'dataset/train',transform=transforms)
    testset1 = datasets.ImageFolder(root=r'dataset/test1',transform=transforms)
    testset2 = datasets.ImageFolder(root=r'dataset/test2',transform=transforms)

    print(f"训练集图片数量: {len(trainset)}")
    print(f"测试集1图片数量: {len(testset1)}")
    print(f"测试集2图片数量: {len(testset2)}")
    #创建数据加载器
    #shuffle=True表示每个epoch都会打乱数据顺序
    # pin_memory=True表示将数据加载到固定内存中以加速GPU训练
    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    test_loader1 = DataLoader(testset1, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    test_loader2 = DataLoader(testset2, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

    #创建网络（在GPU上面进行训练
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = mixed_net().to(device)
    
    #打印网络信息
    summary(net, input_size=(1, 3, 64, 64), device=device)
    print(f'标签对应的ID: {trainset.class_to_idx}')

    #设置优化器、损失函数
    criterion = nn.CrossEntropyLoss()#交叉熵损失函数
    optimizer =optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    # optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-4)
    #ADAM优化器自适应学习率，收敛相比于SGD可能会更快
    
    #开始训练

    print("Start")
    best_performance = {
        'avg_acc': 0,
        'min_acc': 0,
        'epoch': 0,
        'c1': 0,
        'c2': 0
    }
    for epoch in range(EPOCH):
        train_loss = 0.0
        #print(epoch)
        
        for batch_id, (datas, labels) in enumerate(train_loader):
            datas, labels = datas.to(device), labels.to(device)  # 数据移到GPU
    
            optimizer.zero_grad()    # 清零梯度
            outputs = net(datas)     # 前向传播
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()          # 反向传播
            optimizer.step()         # 更新参数
    
            train_loss += loss.item()  # 累积损失

            if epoch > 50 and (epoch + 1) % 10 == 0:#验证模型效果并进行保存
                os.makedirs("pth", exist_ok=True)
                PATH = "pth/modeltemp.pth"          
                torch.save(net.state_dict(), PATH)  #保存当前模型
                model = mixed_net()                 #创建新模型的实例
                model.load_state_dict(torch.load(PATH))     #加载模型参数
                model.eval()                        #设置为评估模式
                model.to(device)                        #将模型移到GPU上

                #限定保存条件
                correct1 = 0
                correct2 = 0
                total1 = 0
                total2 = 0

                #分别测试两个数据集
                with torch.no_grad():   #关闭梯度计算
                    # 测试集1
                    for i ,(datas1, labels1) in enumerate(test_loader1):
                        datas1, labels1 = datas1.to(device), labels1.to(device)
                        output_test1 = model(datas1)    #将测试数据放入当前模型之中，获得模型的预测结果
                        _, predicted1 = torch.max(output_test1.data, dim=1)     #从预测分数中找出每个样本的预测概率和类别
                        total1 += predicted1.size(0)    #计算总数
                        correct1 += (predicted1 == labels1).sum()#计算正确率总和
                    # 测试集2
                    for i ,(datas2, labels2) in enumerate(test_loader2):
                        datas2, labels2 = datas2.to(device), labels2.to(device)
                        output_test2 = model(datas2)
                        _, predicted2 = torch.max(output_test2.data, dim=1)
                        total2 += predicted2.size(0)
                        correct2 += (predicted2 == labels2).sum()

                    #计算准确率
                    c1 = correct1 / total1 * 100
                    c2 = correct2 / total2 * 100
                    avg_acc = (c1 + c2) / 2
                    min_acc = min(c1, c2)
                    
                    #打印消息
                    print(f"epoch:{epoch + 1}\tbatch_id:{batch_id + 1}\t"
                          f"average_loss:{(train_loss / len(train_loader.dataset)):.5f}\t"
                          f"correct1:{c1:.2f}%\tcorrect2:{c2:.2f}%\t"
                          f"avg:{avg_acc:.2f}%\tmin:{min_acc:.2f}%")
                    
                    # 改进的保存策略：综合考虑两个测试集
                    should_save = False
                    save_reason = ""
                    
                    # 条件：平均准确率提升且最低准确率不低于95%
                    if avg_acc > best_performance['avg_acc'] and min_acc >= 95:
                        should_save = True
                        save_reason = "avg_improved_with_good_min"
                    # 或者最低准确率显著提升
                    elif min_acc > best_performance['min_acc'] + 1:  # 至少提升1%
                        should_save = True
                        save_reason = "min_significantly_improved"
                    
                    if should_save:
                        # 更新最佳性能记录
                        best_performance.update({
                            'avg_acc': avg_acc,
                            'min_acc': min_acc,
                            'epoch': epoch + 1,
                            'c1': c1,
                            'c2': c2
                        })
                        
                        MAX_PATH = f"pth/model2_balanced_avg{avg_acc:.1f}_min{min_acc:.1f}_e{epoch+1}.pth"
                        print(f"save {MAX_PATH} - {save_reason}")
                        torch.save(net.state_dict(), MAX_PATH)

