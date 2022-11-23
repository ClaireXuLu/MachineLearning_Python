# ResNet : nested function
# 保证层数增加一定不会使结果更差，也就是函数的模型复杂度增加同时包含了之前的模型
# ResNet块
# f(x) = x + g(x)
# 残差块使得更深的网络更容易训练
# 可以有效解决低端的权重的梯度消失的问题
from utils import *
from torch.nn import functional as F
import torch
import torchvision
from torchvision import transforms
from torch.utils import data
import time
from torch import nn
import matplotlib.pyplot as plt

class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, stride=1):
        """_summary_

        Args:
            input_channels (_type_): input channel
            num_channels (_type_): output channel
            use_1x1conv (bool, optional): flag whether to expand num_channels. Defaults to False.
            stride (int, optional): . Defaults to 1.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)


def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            # 高宽减半
            blk.append(Residual(input_channels, num_channels, use_1x1conv=True, stride=2))
        else:
            # 高宽不变
            blk.append(Residual(num_channels, num_channels))
    
    return nn.Sequential(*blk)



def main():
    b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                       nn.BatchNorm2d(64), nn.ReLU(),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    b2 = resnet_block(64, 64, 2, first_block=True) # 高宽不变
    b3 = resnet_block(64, 128, 2) # 高宽减半，通道增倍
    b4 = resnet_block(128, 256, 2)     
    b5 = resnet_block(256, 512, 2)    
    net = nn.Sequential(b1, b2, b3, b4, b5, nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(512, 10))   
    
    X = torch.randn(1, 1, 224, 224)
    for layer in net:
        X = layer(X)
        print(layer.__class__.__name__, 'output shape:\t', X.shape)
    
    lr, num_epochs, batch_size = 0.05, 10, 256
    train_iter, test_iter = load_data(batch_size)
    train(net, train_iter, test_iter, lr, num_epochs, get_device())
    



if __name__ == '__main__':
    main()

