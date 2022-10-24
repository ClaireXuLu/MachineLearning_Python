# 七、CNN卷积神经网络

CNN是广泛应用在计算机视觉领域的网络，其属于神经网络中的一个类别。

## 1.卷积层

这里用吴恩达课程中的垂直边缘检测的例子作为卷积运算的范例，其实很简单，就是kernel和对应的区域相乘求和，之后得到的矩阵再输入激活函数。

其实可以发现相比较于之前提到的神经网络，卷积运算并没有把特征展开，而是直接利用二维核移动着与各个区域的特征进行运算，这种方式是和图片相适应的模型构建，因为图片往往size相对很大，如果用全连接层网络的话，需要配置的参数会很多，训练会很慢。但卷积运算就解决了这个问题，共享权重且稀疏连接使得训练参数大幅度降低。

另外，有时为了边缘的特征不浪费以及数据规模不缩小，会对初始数据进行padding操作。基于是否padding我们把卷积操作分为 volid convolution 和 same convolution

![What is Padding in Neural Network? - GeeksforGeeks](https://media.geeksforgeeks.org/wp-content/uploads/20220302180104/Group16.jpg)



通常一个卷积层确定输出规模需要确定以下几个元素：

1. 特征维度 h * w * c（h是高度，w是宽度，c是频道数目）
2. 卷积核大小 f*f
3. padding 数目 
4. 移动的步长s
5. kernel的数目 k

假设输入的特征为$n_h*n_w*n_c*m$, 则输出的特征维度为$(\frac{n_h+2p-f}{s}+1)*\frac{n_w+2p-f}{s}+1)*n_k*m$



![image-20221021170945699](https://s2.loli.net/2022/10/21/mYyOjhgflCbk63d.png)

![image-20221021172949034](https://s2.loli.net/2022/10/21/5ekmlTarCxUYsD8.png)



## 2.池化层

通常CNN的网络架构里除了卷积层还需要池化层，为的是缩小feature规模，加快训练速度，常用的池化层方法有max-pooling, average-pooling。

![17.5 池化的前向计算与反向传播- AI-EDU](https://microsoft.github.io/ai-edu/%E5%9F%BA%E7%A1%80%E6%95%99%E7%A8%8B/A2-%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%9F%BA%E6%9C%AC%E5%8E%9F%E7%90%86/%E7%AC%AC8%E6%AD%A5%20-%20%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C/img/17/pooling.png)

## 3.基本模型结构

一个常见的卷积神经网络包括卷积层，池化层，全连接层。

![image-20221024091123079](https://s2.loli.net/2022/10/24/uMsNz8YC9hLwQJE.png)
