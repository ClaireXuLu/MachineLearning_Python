import torch
import torchvision
from torchvision import transforms
from torch.utils import data
import time
from torch import nn
import matplotlib.pyplot as plt



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