import torch
from torch import nn, optim
import numpy as np
from PIL import Image
import re


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
        # 全连接层
        self.fc = nn.Sequential(  
            nn.Linear(400, 120),
            nn.Linear(120, 84),
            nn.Linear(84, n_class)
        )

    def forward(self, x):
        out = self.conv(x)  # out shape(batch,16,5,5)
        out = out.view(out.size(0), -1)  # out shape(batch,400)
        out = self.fc(out)  # out shape(batch,10)
        return out


def identify(input_image, load_mode):
    #  要识别的图片
    #input_image = 'out/R/B1.png'

    cnn = CNN(1, 10)

    # 加载模型-digital是数码管模型，jiugongge是九宫格识别模型
    # 注意，数字1识别效果不行，容易和8混淆
    #ckpt = torch.load('./mode/digital.tar')
    ckpt = torch.load(load_mode)

    # 参数加载到指定模型cnn
    cnn.load_state_dict(ckpt)

    im = Image.open(input_image).resize((28, 28))  # 取图片数据
    im = im.convert('L')  # 灰度图
    im_data = np.array(im)
    im_data = torch.from_numpy(im_data).float()

    im_data = im_data.view(1, 1, 28, 28)
    # 图片数据经过处理之后丢入CNN模型之中，求出最大的可能性的值出来
    # 这个最大可能性就是结果
    out = cnn(im_data)
    _, pred = torch.max(out, 1)
    
    # print("%s"%str(re.findall(r"\d",str(pred))))
    # print(im_data)
    #print('预测为:数字{}。'.format(pred))

    # 利用正则表达式把数字从数据里面读出来
    return re.findall(r"\d",str(pred))


#丢进去的图片数据需要提前二值化
#第一个参数为需要进行处理的图片，第二个参数为输出模型
print(identify("./pic/test.png","./mode/output.tar"))