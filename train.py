import torch
import torchvision 
from tqdm import tqdm
import matplotlib.pyplot as plt
from net import Net
from torchvision.transforms import Compose, Normalize, Resize, RandomAdjustSharpness, ToTensor
import os
import warnings 
warnings.filterwarnings("ignore")


def train(trainDataLoader, testDataLoader, net, lr, history):
    lossF = torch.nn.CrossEntropyLoss().cpu()  # 目标函数，也叫损失函数，模型训练就是为了让目标函数的值越来越小，趋近于0
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-3)   # 学习器，torch库封装好的，使用梯度下降算法来更新 Net() 的参数
    
    """
    训练部分
    """
    #构建tqdm进度条
    processBar = tqdm(trainDataLoader, unit = 'step')
    
    #打开网络的训练模式
    net.train()
    #开始对训练集的DataLoader进行迭代
    for _, (data, labels) in enumerate(processBar):
        data = data.cpu()
        labels = labels.cpu()
        
        net.zero_grad()  # 清空模型的梯度
        outputs = net(data)   # 对模型进行前向推理
        loss = lossF(outputs, labels)  # 计算本轮推理的Loss值
        
        # 计算本轮推理的准确率
        predictions = torch.argmax(torch.softmax(outputs, dim=1), dim=-1)
        accuracy = torch.sum(predictions == labels) / labels.shape[0]
        
        loss.backward()  # 进行反向传播求出模型参数的梯度
        optimizer.step()  # 使用迭代器更新模型权重

        # 将本step结果进行可视化处理
        processBar.set_description("Train Loss: %.4f, Acc: %.4f" % (loss.item(), accuracy.item()))
    
    processBar.close()
    
    """
    测试部分
    """
    processBar = tqdm(testDataLoader, unit = 'step')

    #构造临时变量
    correct, totalLoss = 0, 0
    
    #关闭模型的训练状态 或者使用 net.train(False) 也行
    net.eval()
    #对测试集的DataLoader进行迭代
    for _, (data, labels) in enumerate(processBar):
        data = data.cpu()
        labels = labels.cpu()

        outputs = net(data)
        loss = lossF(outputs, labels)
        predictions = torch.argmax(torch.softmax(outputs, dim=1), dim=-1)
        
        totalLoss += loss
        correct += torch.sum(predictions == labels)
        
    testAccuracy = correct / (BATCH_SIZE * len(testDataLoader))
    testLoss = totalLoss / len(testDataLoader)
    
    history['Test Loss'].append(testLoss.item())
    history['Test Accuracy'].append(testAccuracy.item())
     
    print("Test Loss: %.4f, Test Acc: %.4f" % (testLoss.item(), testAccuracy.item()), flush=True)
    processBar.close()
    
    """
    保存模型
    """
    torch.save(net, './model.pth')
    
    
    
def getLoader():
    transforms = Compose([  # 保证输入时PLI格式的图片
        ToTensor(),
        Normalize(mean = [0.5], std = [0.5]),
        Resize((64, 64), antialias=None),
        RandomAdjustSharpness(sharpness_factor=10, p=1)
    ])
    
    #下载训练集和测试集
    trainData = torchvision.datasets.MNIST(mnist_data_path, train=True, transform=transforms, download=True)
    testData = torchvision.datasets.MNIST(mnist_data_path, train=False, transform=transforms, download=True) 

    #构建数据集和测试集的DataLoader
    trainDataLoader = torch.utils.data.DataLoader(dataset=trainData, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    testDataLoader = torch.utils.data.DataLoader(dataset=testData, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)

    return trainDataLoader, testDataLoader



if __name__ == '__main__':
    # 切换到当前文件所在的目录
    os.chdir("/home/Guanjq/Work/DigitClassifier/")   
    
    mnist_data_path = './mnist_data/'  #数据集下载后保存的目录
    net = Net().cpu()         # 把模型放在cpu上面跑
    
    history = {'Test Loss':[], 'Test Accuracy':[]}  # 存储训练的数据
    
    BATCH_SIZE = 16
    EPOCHS = 5  # 总的循环, 学习4轮的效果已经很好了
    learning_rate = 1e-4  # 学习率
    
    trainDataLoader, testDataLoader = getLoader()
    
    for epoch in range(1,EPOCHS + 1):
        train(trainDataLoader, testDataLoader, net, learning_rate, history)
        learning_rate *= 0.8  # 每次学习率衰减0.8倍

    #对测试Loss进行可视化
    plt.plot(history['Test Loss'],label = 'Test Loss')
    plt.legend(loc='best')
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig("Loss.png")
    plt.show()

    #对测试准确率进行可视化
    plt.plot(history['Test Accuracy'],color = 'red',label = 'Test Accuracy')
    plt.legend(loc='best')
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.savefig("Accuracy.png")
    plt.show()

