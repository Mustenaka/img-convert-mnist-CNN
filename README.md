# CNN:PNG-CONVERT-MNIST

本项目旨在于使用自己预处理的图片创建mnist格式的数据集并且通过卷积神经网络（CNN）训练出自己的简单的数字识别器。

数据集的转换参考自GITHUB项目：convert-images-to-mnist-format，并且进行了稍微的修改（原版代码基于py2.7且有部分内容较老，部分运行还是基于bash的，无法直接在WINDOWS上面运行，故修改为WINDOWS可以运行的版本）。



### 如何使用？

首先请将你需要的数据预处理为28*28尺寸的二值化图片，标准的mnist格式也是如此，或者是处理成灰度图的格式，请尽量自主排除干扰信息，文件格式默认为png，可以修改convert-images-to-mnist-format.py的第9行代码。

将你自己预处理好的图片分别存放在data目录下的training-images目录和test-images目录，并且以再该目录下创建你所需要分类个数的子文件夹，子文件夹的命名就是图片的标签。![1-1](.\1-1.png)
![1-1](https://github.com/Mustenaka/img-convert-mnist-CNN/blob/master/1-1.png)


接着运行convert-images-to-mnist-format.py生成以下文件。

1. 训练集数据
2. 训练集标签数据
3. 测试集数据
4. 测试集标签数据

将上诉生成的四个文件移动至mnist\MNIST\raw的路径下，并且可以运行train_gpu.py训练CNN模型。

如果你没有数据，可以将train_gpu.py的，第33行中的False修改为True改为自动下载mnist数据，如此则不需要前置运行convert-images-to-mnist-format.py

训练次数的设置在train_gpu.py的31行num_epoches，建议数字为5~25之间，超过30会可能会出现过拟合的情况



运行结果将会在mode中生成output.tar的模型文件，可以使用feature.py运行进行验证测试，只需要修改以下identify函数的传递参数即可，如：（第一个参数为需要测试的图片，第二个为采用的模型）

print(identify("./pic/test.png","./mode/output.tar"))



### 运行条件：

本人使用python3.7，需要调用到以下的库，需要提前安装CUDA10：

pillow
opencv-python
torch==1.2.0
torchvision==0.4.0
scipy



如果允许convert-images-to-mnist-format.py则还需要给电脑安装gzip，可以在这个页面下载WINDOWS版本http://gnuwin32.sourceforge.net/packages/gzip.htm

