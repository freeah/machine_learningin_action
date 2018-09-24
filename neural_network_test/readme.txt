首先理解前向传播和反向传播
[http://www.iro.umontreal.ca/~pift6266/A06/refs/backprop_old.pdf（反向传播论文）](http://www.iro.umontreal.ca/~pift6266/A06/refs/backprop_old.pdf（反向传播论文）)
代码参考：
[http://neuralnetworksanddeeplearning.com/chap1.html](http://neuralnetworksanddeeplearning.com/chap1.html)

反向传播参考链接：
[https://blog.csdn.net/bitcarmanlee/article/details/78819025](https://blog.csdn.net/bitcarmanlee/article/details/78819025)
[http://www.sohu.com/a/229530883_610300](http://www.sohu.com/a/229530883_610300)

（MNIST数据集）：
训练集60000个   测试集10000个   每张图片大小28*28=784像素

将每张图像数组转换成784行一列的向量，进行训练，训练的网络有三层
第一层 x有784行1列   w有40行784列  b有40行一列   生成z1有40行一列，使用激活函数输出，即下一层的输入
第二层x有40行一列，w有10行40列，b有10行一列，生成z2有10行1列，使用激活函数，输出最后一层

网络的输出层包含10个神经元。如果第一个神经元发射，即输出≈1，那么这将表明网络认为该数字是0.如果第二个神经元发射，则表明网络认为该数字为1。依此类推。更准确地说，我们将输出神经元从0到9编号，并找出哪个神经元具有最高的激活值。如果那个神经元是神经元数6，那么我们的网络将猜测输入数字是6.等等其他输出神经元。

步骤：
1.定义一个网络
2、找出训练集、测试集
3、训练数据，训练出合适的w和b
4、使用测试数据验证准确率
