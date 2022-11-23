import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
import torchvision
from torchvision import transforms
from torch.utils import data
import time
from torch import nn
import matplotlib.pyplot as plt
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,7'


def get_device(i=0):
    if torch.cuda.device_count() >= (i+1):
        return torch.device(f'cuda:{i}')
    else:
        return torch.device('cpu')


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

def load_data( batch_size, img_size=(224, 224)):
    trans = [transforms.Resize(img_size),transforms.ToTensor()]
    trans =transforms.Compose(trans)

    mnist_train = torchvision.datasets.FashionMNIST(root='./data', train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root='./data', train=False, transform=trans, download=True)

    train_iter = data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
    test_iter = data.DataLoader(mnist_test, batch_size=batch_size)
    return train_iter, test_iter


def accuracy_score(net, data_iter, device):
    net.eval()
    acc_num , total_num = 0, 0
    for X, y in data_iter:
        X = X.to(device)
        y = y.to(device)
        y_hat = net(X)
        y_pred = torch.argmax(y_hat, axis=1)
        acc_num += torch.sum(y_pred==y).item()
        total_num += y.numel()
    return acc_num / total_num


def resnet18(num_classes, in_channels=1):
    """稍加修改的 ResNet-18 模型。"""
    def resnet_block(in_channels, out_channels, num_residuals,
                     first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(Residual(in_channels, out_channels, use_1x1conv=True, strides=2))
            else:
                blk.append(Residual(out_channels, out_channels))
        return nn.Sequential(*blk)

    net = nn.Sequential(
        nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64), nn.ReLU())
    net.add_module("resnet_block1", resnet_block(64, 64, 2, first_block=True))
    net.add_module("resnet_block2", resnet_block(64, 128, 2))
    net.add_module("resnet_block3", resnet_block(128, 256, 2))
    net.add_module("resnet_block4", resnet_block(256, 512, 2))
    net.add_module("global_avg_pool", nn.AdaptiveAvgPool2d((1, 1)))
    net.add_module("fc",
                   nn.Sequential(nn.Flatten(), nn.Linear(512, num_classes)))
    return net


def train(net, train_iter, test_iter, num_gpus, num_epochs, lr):

    print('training on', list(range(torch.cuda.device_count())))
    def init_weights(m):
        if type(m) in [nn.Linear, nn.Conv2d]:
            nn.init.normal_(m.weight, std=0.1)
    net.apply(init_weights)
    
    net = nn.DataParallel(net)
    net = net.cuda()
    trainer = torch.optim.SGD(net.parameters(), lr)
    loss = nn.CrossEntropyLoss()
    train_ls, test_ls = [], []
    start_time = time.time()
    for epoch in range(num_epochs):
        net.train()
        for X, y in train_iter:
            trainer.zero_grad()
            X, y = X.cuda(), y.cuda()
            l = loss(net(X), y)
            l.backward()
            trainer.step()
        with torch.no_grad():
            train_acc = accuracy_score(net, train_iter, get_device())
            test_acc = accuracy_score(net, test_iter, get_device())
            train_ls.append(train_acc)
            test_ls.append(test_acc)
            print(f'epoch {epoch+1}, train_acc {train_acc:.3f}, test_acc {test_acc:.3f}')
    end_time = time.time()
    print(f'{(end_time-start_time)/num_epochs:.2f}s/epoch')
    plt.plot(range(1, num_epochs+1), train_ls, range(1, num_epochs+1), test_ls)
    plt.show()
    return train_ls, test_ls

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
    batch_size , lr, num_gpus, num_epochs= 128, 0.1, 3, 10
    train_iter, test_iter = load_data(batch_size)
    train(net, train_iter, test_iter, num_gpus, num_epochs, lr)
    

    
      
if __name__ == '__main__':
    main()
