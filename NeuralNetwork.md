# 六、神经网络

神经网络是由感知机中的元件作为基本组成元件构造的多层网络结构，其模型的构建计算过程y=f(x) 我们通常称为前向传播，梯度下降过程中计算导数通常称为反向传播。

![统计学习方法：感知机- 桂。 - 博客园](https://images2015.cnblogs.com/blog/1085343/201704/1085343-20170416123050787-138884231.png)

<center> 
    figure 1 感知机模型
</center>





![卷积神经网络CNN总结- Madcola - 博客园](https://images2015.cnblogs.com/blog/1093303/201704/1093303-20170430194200912-687300437.jpg)

<center>
    figure 2 神经网络模型
</center>





## 前向传播

在写明计算公式前先声明一些变量定义。输出求和节点的向量记为z，经过激活函数的记为a，此层的权重记为W，偏差记为b，其有如下关系：
$$
a^{[l]} = \sigma(z^{[l]})=\sigma(W^{[l]}a^[[l-1]]+b^{[l]})
$$
其中$a^{[l]}$ 向量的维度为$n^{[l]} * 1 $ ，$W^{[l]}$ 的维度为 $n^{[l]} * n^{[l-1]}$  , $b^{[l]}$ 向量的维度为$n^{[l]} * 1 $ 

假定输入层为第0层，其输出的变量定义为$a^{[0]} =x$ 

上述这个公式的计算过程其实就是前向传播算法。直到最后一层输出层我们计算出最终的$y=a^{output}$ 

## 反向传播

模型构建好了，此时我们需要对其构建一个损失函数，然后进行梯度下降，此过程中有大量的参数W, b。那么如何计算呢? 我们利用链式法则！假设损失函数为J，

假设我们已知$\frac{\partial J}{\partial {a^{[l+1]}}}$ , 因为 $a^{[l+1]} = \sigma(z^{[l+1]})$ ,则有：
$$
\frac{\partial J}{\partial {z^{[l+1]}}} = \frac{\partial J}{\partial {a^{[l+1]}}} * \frac{\partial {a^{[l+1]}}}{\partial {z^{[l+1]}}}
=\frac{\partial J}{\partial {a^{[l+1]}}} * \sigma'(z^{[l+1]})
$$


又因为 $z^{[l+1]}=W^{[l+1]}a^[[l]]+b^{[l+1]} $
$$
\frac{\partial J}{\partial {W^{[l+1]}}} 
= \frac{\partial J}{\partial {z^{[l+1]}}} * \frac{\partial {z^{[l+1]}}}{\partial {W^{[l+1]}}}
= \frac{\partial J}{\partial {a^{[l+1]}}} * \sigma'(z^{[l+1]}) * (a^{[l]})^T
\\
\frac{\partial J}{\partial {b^{[l+1]}}} 
= \frac{\partial J}{\partial {z^{[l+1]}}} * \frac{\partial {z^{[l+1]}}}{\partial {b^{[l+1]}}}
= \frac{\partial J}{\partial {a^{[l+1]}}} * \sigma'(z^{[l+1]}) 
\\
\frac{\partial J}{\partial {a^{[l]}}} 
= \frac{\partial J}{\partial {z^{[l+1]}}} * \frac{\partial {z^{[l+1]}}}{\partial {a^{[l]}}}
= (W^{[l+1]})^T * \frac{\partial J}{\partial {a^{[l+1]}}}* \sigma'(z^{[l+1]})
$$
通过上述的反向传播计算过程即可计算出所有权重的偏导数。

## 总结

这里利用吴恩达课程中一张图更好的理解前向传播和反向传播的过程。

![image-20221020153855604](https://s2.loli.net/2022/10/21/4Sc6sFERJZVozU5.png)