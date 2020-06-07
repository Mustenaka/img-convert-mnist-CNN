import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from skimage import io, data, transform

import numpy as np
import os

'''
训练CNN模型使用的文件
基本上就是模板CNN结构，效果对于类别种类很小的数据处理效果很好，尤其MNIST，基本两轮都可以ACC达到0.97
训练次数：num_epoches 建议设置在3~20这个范围，超过会发送过拟合，识别效率反而不好
----------------------------
卷积神经网络：
卷积层，池化层，激活，全连接等，可以直接看注释
----------------------------
训练数据和测试数据大概占比7：3
使用torch.save(cnn.state_dict())的方式保存模型参数
----------------------------
已经经过二值化处理的数据相对而言比较好处理
在进行这个训练算法之前请必须使用convert-images-to-mnist-format.py
这个模块将图像数据提前处理成mnist的数据格式
'''

batch_size = 200    # 分批训练数据、每批数据量
learning_rate = 1e-2    # 学习率
num_epoches = 25        # 训练次数
posPath = "./mode/output.tar"  # 生成训练模型路径
DOWNLOAD_MNIST = False    # 是否网上下载数据


# 目标大小
MNIST_SIZE = 28


def translate(image_path):
    # 读入图片并变成灰色
    img = io.imread(image_path)
    # 缩小到28*28
    translated_img = transform.resize(img, (MNIST_SIZE, MNIST_SIZE))
    # 变成1*784的一维数组
    flatten_img = np.reshape(translated_img, 784)
    # mnist数据集中1代表黑，0代表白
    result = np.array([1 - flatten_img])
    # 返回该图的所代表的向量
    return result


# mnist数据集下载 ×
# 利用其创建mnist数据格式而已
if not(os.path.exists('./mnist/')) or not os.listdir('./mnist/'):
    # not mnist dir or mnist is empyt dir
    DOWNLOAD_MNIST = True


# 将mnist数据格式转换为tensor
train_dataset = datasets.MNIST(
    root='./mnist',
    train=True,  # download train data
    transform=transforms.ToTensor(),
    download=DOWNLOAD_MNIST
)
test_dataset = datasets.MNIST(
    root='./mnist',
    train=True,  # download test data
    transform=transforms.ToTensor(),
    download=DOWNLOAD_MNIST
)

# 该接口主要用来将自定义的数据读取接口的输出或者PyTorch已有的数据读取接口的输入
# 按照batch size封装成Tensor，后续只需要再包装成Variable即可作为模型的输入
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True)  # shuffle 是否打乱加载数据
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


class CNN(nn.Module):
    def __init__(self, in_dim, n_class):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, 6, kernel_size=3, stride=1, padding=1),
            # input shape(1*28*28),(28+1*2-3)/1+1=28 卷积后输出（6*28*28）
            # 输出图像大小计算公式:(n*n像素的图）(n+2p-k)/s+1
            nn.ReLU(True),        # 激活函数
            nn.MaxPool2d(2, 2),    # 28/2=14 池化后（6*14*14）
            # (14-5)/1+1=10 卷积后（16*10*10）
            nn.Conv2d(6, 16, 5, stride=1, padding=0),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)  # 池化后（16*5*5）=400，the input of full connection
        )
        self.fc = nn.Sequential(  # full connection layers.
            nn.Linear(400, 120),
            nn.Linear(120, 84),
            nn.Linear(84, n_class)
        )

    def forward(self, x):
        out = self.conv(x)  # out shape(batch,16,5,5)
        out = out.view(out.size(0), -1)  # out shape(batch,400)
        out = self.fc(out)  # out shape(batch,10)
        return out


cnn = CNN(1, 10)
print(cnn)

if torch.cuda.is_available():  # 是否可用GPU计算
    cnn = cnn.cuda()  # 转换成可用GPU计算的模型

criterion = nn.CrossEntropyLoss()  # 多分类用的交叉熵损失函数
optimizer = optim.Adam(cnn.parameters(), lr=learning_rate)
# 常用优化方法有
# 1.Stochastic Gradient Descent (SGD)
# 2.Momentum
# 3.AdaGrad
# 4.RMSProp
# 5.Adam (momentum+adaGrad)   效果较好

for epoch in range(num_epoches):
    print('epoch{}'.format(epoch+1))
    print('*'*10)
    running_loss = 0.0
    running_acc = 0.0
    # 训练
    for i, data in enumerate(train_loader, 1):
        img, label = data
        #  判断是否可以使用GPU，若可以则将数据转化为GPU可以处理的格式。
        if torch.cuda.is_available():
            img = Variable(img).cuda()
            label = Variable(label).cuda()
        else:
            img = Variable(img)
            label = Variable(label)
        out = cnn(img)
        loss = criterion(out, label)
        running_loss += loss.item() * label.size(0)
        _, pred = torch.max(out, 1)
        num_correct = (pred == label).sum()
        accuracy = (pred == label).float().mean()
        running_acc += num_correct.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('Finish {} epoch,Loss:{:.6f},Acc:{:.6f}'.format(
        epoch+1, running_loss /
        (len(train_dataset)), running_acc/len(train_dataset)
    ))

    # 测试
    cnn.eval()  # eval()时，模型会自动把BN和DropOut固定住，不会取平均，而是用训练好的值
    eval_loss = 0
    eval_acc = 0
    for i, data in enumerate(test_loader, 1):
        img, label = data
        # 判断是否可以使用GPU，若可以则将数据转化为GPU可以处理的格式。
        if torch.cuda.is_available():
            img = Variable(img).cuda()
            label = Variable(label).cuda()
        else:
            img = Variable(img)
            label = Variable(label)

        out = cnn(img)
        loss = criterion(out, label)
        eval_loss += loss.item() * label.size(0)
        _, pred = torch.max(out, 1)
        num_correct = (pred == label).sum()
        accuracy = (pred == label).float().mean()
        eval_acc += num_correct.item()

    print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
        test_dataset)), eval_acc/len(test_dataset)))

torch.save(cnn.state_dict(), posPath)
