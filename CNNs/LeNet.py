import torch
from torch import nn
from torchvision import transforms
import torchvision
from torch.utils import data

# 自定义层
class Reshape(nn.Module):
    def forward(self, x):
        return x.view(-1, 1, 28, 28)

        
def get_device(i=0):
    if torch.cuda.device_count() >= (i+1):
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def accuracy_score(net, dataset_iter, device):
    net.eval()
    acc_num = 0
    total_num = 0
    for X, y in dataset_iter:
        X = X.to(device)
        y = y.to(device)
        y_hat = net(X)
        y_pred = torch.argmax(y_hat, axis=1)
        acc_num += torch.sum(y_pred==y).item()
        total_num += y.numel()
    return acc_num / total_num


def train(net, train_iter, test_iter, num_epochs, lr, device):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    num_batches = len(train_iter)
    train_acc_ls, test_acc_ls = [], []
    for epoch in range(num_epochs):
        net.train()
        for i, (X, y) in enumerate(train_iter):
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
        with torch.no_grad():
            train_acc = accuracy_score(net, train_iter, device)
            test_acc = accuracy_score(net, test_iter, device)
            train_acc_ls.append(train_acc)
            test_acc_ls.append(test_acc)
            print(f'epoch {epoch+1}, train_acc {train_acc:.4f}, test_acc {test_acc:.4f}')

def main():
    batch_size= 256
    num_epochs, lr = 5, 0.9
    net = nn.Sequential(
                Reshape(),
                nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
                nn.AvgPool2d(kernel_size=2, stride=2),
                nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
                nn.AvgPool2d(kernel_size=2, stride=2),
                nn.Flatten(),
                nn.Linear(16*5*5, 120), nn.Sigmoid(),
                nn.Linear(120, 84), nn.Sigmoid(),
                nn.Linear(84, 10))
    print(net)

    # 加载数据在fashion Mnist数据集测试lenet的效果
    trans = transforms.ToTensor()
    mnist_train = torchvision.datasets.FashionMNIST(root='./data', train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root='./data', train=False, transform=trans, download=True)
    train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=2)
    test_iter = data.DataLoader(mnist_test, batch_size, shuffle=True, num_workers=2)
    
    train(net, train_iter, test_iter, num_epochs, lr, device=get_device())

if __name__=='__main__':
    main()