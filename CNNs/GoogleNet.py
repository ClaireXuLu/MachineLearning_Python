# 超过100层的卷积网络
# 卷积核的选择问题的解决，Inception块
# Inception的高宽不变，通过降低通道数来控制模型复杂度
# Inception使得通道变成加法累加而不是乘法，即增加模型的丰富度，又减小了模型计算量
import torch
from torch import nn
from torchvision import transforms
import torchvision
from torch.utils import data
import matplotlib.pyplot as plt
from torch.nn import functional as F
import time


class Inception(nn.Module):
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        return torch.cat((p1, p2, p3, p4), dim=1)


def get_device(i=0):
    if torch.cuda.device_count() >= (i+1):
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def load_data( batch_size, img_size=(224, 224)):
    trans = [transforms.Resize(img_size),transforms.ToTensor()]
    trans =transforms.Compose(trans)

    mnist_train = torchvision.datasets.FashionMNIST(root='./data', train=True, transform=trans, download=False)
    mnist_test = torchvision.datasets.FashionMNIST(root='./data', train=False, transform=trans, download=False)

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


def train(net, train_iter, test_iter, lr, num_epochs, device):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
        
    net.apply(init_weights)
    net.to(device)
    print('training on', device)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    train_ls, test_ls = [], []
    start_time = time.time()
    for epoch in range(num_epochs):
        net.train()
        for X, y in train_iter:
            optimizer.zero_grad()
            X = X.to(device)
            y = y.to(device)
            l = loss(net(X), y)
            l.backward()
            optimizer.step()
        with torch.no_grad():
            train_acc = accuracy_score(net, train_iter, device)
            test_acc = accuracy_score(net, test_iter, device)
            train_ls.append(train_acc)
            test_ls.append(test_acc)
            print(f'epoch {epoch+1}, train_acc {train_acc:.3f}, test_acc {test_acc:.3f}')
    end_time = time.time()
    print(f'{(end_time-start_time)/num_epochs:.2f}s/epoch')
    plt.plot(range(1, num_epochs+1), train_ls, range(1, num_epochs+1), test_ls)
    plt.show()
    return train_ls, test_ls


def main():
    lr, num_epochs, batch_size = 0.1, 10, 128
    b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                   nn.ReLU(), nn.MaxPool2d(kernel_size=3, stride=2,
                                           padding=1))

    b2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1), nn.ReLU(),
                    nn.Conv2d(64, 192, kernel_size=3, padding=1),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    b3 = nn.Sequential(Inception(192, 64, (96, 128), (16, 32), 32),
                    Inception(256, 128, (128, 192), (32, 96), 64),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    b4 = nn.Sequential(Inception(480, 192, (96, 208), (16, 48), 64),
                    Inception(512, 160, (112, 224), (24, 64), 64),
                    Inception(512, 128, (128, 256), (24, 64), 64),
                    Inception(512, 112, (144, 288), (32, 64), 64),
                    Inception(528, 256, (160, 320), (32, 128), 128),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    b5 = nn.Sequential(Inception(832, 256, (160, 320), (32, 128), 128),
                    Inception(832, 384, (192, 384), (48, 128), 128),
                    nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())

    net = nn.Sequential(b1, b2, b3, b4, b5, nn.Linear(1024, 10))


    train_iter, test_iter = load_data(batch_size, img_size=(96, 96))
    train_acc, test_acc = train(net, train_iter, test_iter, lr, num_epochs, get_device())



if __name__ == '__main__':
    main()