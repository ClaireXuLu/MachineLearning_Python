# PCA

## 1. 算法原理

pca是特征降维的经典算法之一，特征降维直白来说就是降低样本的特征维度同时又不希望丢失过多信息。如何将这个目标转换为数学问题。可以从两个角度出发：

一是使得降维后的向量和原始向量之间的误差和最小化，基于最小投影距离：
$$
\mathop{argmin}_W　 (-tr(W^TXX^TW))  　 s.t. W^TW=I
$$
二是使得样本点经过投影后还能区分的开，方差大，基于最大投影方差：
$$
\mathop{argmax}_W (tr(W^TXX^TW)　s.t. W^TW=I
$$
这里W是 n*k维的向量，n是初始向量维度, k是降维后的维度。具体的推导其实我还是不很清楚，先了解一个算法实现，之后需要我再去仔细理解推导，可以参考这篇博客：[PCA原理总结][https://www.cnblogs.com/pinard/p/6239403.html] 

## 2. 算法流程总结

1. 对所有样本去中心化 $x^{(i)} = x^{(i)} - \frac{1}{m}\sum_{j=1}^m{x^{(j)}}$ 
2. 计算样本的协方差矩阵$X^TX$ 
3. 对协方差矩阵$X^TX$ 进行特征值分解
4. 取出最大的k个特征值对应的特征向量
5. 对每个样本进行特征转换 $z^{(i)} = W^Tx^{(i)}$ 

   或者有时降维并不指定k，而是指定主成分的比重阈值t，选取前k大的特征值满足：
$$
\frac{\sum_{i=1}^k\lambda_i}{\sum_{i=1}^n\lambda_i} \ge t
$$


## 3. python算法实现

```python
import numpy as np

def pca(X, k):
    """dimensionality reduction method

    Args:
        X (ndarray): m * n, m samples, n dims feature
        k (integer): hope to dimension's final ndim number

    Returns:
        Z (ndarray): m * k
    """
    # decentralization
    X = X - np.mean(X, axis=0)
    # calculate covariance matrix of X
    X_cov = np.cov(X, rowvar=False)
    # eigenvalue decomposition of the covariance matrix
    eigvalues, eigvectors = np.linalg.eig(X_cov)
    # pick first-k eigvector as W
    max_eigvalue_index = np.argsort(-eigvalues)[:k]
    W = eigvectors[:, max_eigvalue_index]

    Z = X @ W
    return Z

```

### 3.1 一个实例

```python
X = [[2.5, 2.4],
     [0.5, 0.7],
     [2.2, 2.9],
     [1.9, 2.2],
     [3.1, 3.0],
     [2.3, 2.7],
     [2, 1.6],
     [1, 1.1],
     [1.5, 1.6],
     [1.1, 0.9]]
Z = pca(X, 1)
print(Z)
```

### 3.2 结果

![image-20221018211330024](C:\Users\徐璐\AppData\Roaming\Typora\typora-user-images\image-20221018211330024.png)

























