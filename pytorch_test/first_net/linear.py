#关系拟合(回归)
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
#unsqueeze把一维数据转化成二维数据，torch中只能处理二维数据
x=torch.unsqueeze(torch.linspace(-1,1,100),dim=1)
y=x.pow(2)+0.2*torch.rand(x.size())


#x，y转化成Variable形似，神经网络只能输入Variable
x,y=Variable(x),Variable(y)
# plt.scatter(x.data.numpy(),y.data.numpy())
# plt.show()

#继承Module模块
class Net(torch.nn.Module):
	def __init__(self,n_feature,n_hidden,n_output):
		super(Net,self).__init__()
		self.hidden=torch.nn.Linear(n_feature,n_hidden) #隐藏层  参数表示输入个数和隐藏层的个数
		self.predict=torch.nn.Linear(n_hidden,n_output)  #预测



	def forward(self,x):
		x=F.relu(self.hidden(x))
		x=self.predict(x)
		return x

net=Net(1,10,1)
print(net)		

#优化器优化神经网络参数,传入所有参数
optimizer=torch.optim.SGD(net.parameters(),lr=0.5)
#loss函数，均方差，主要针对线性回归
loss_func=torch.nn.MSELoss()

error_count=0.0
#训练
for t in range(100):
	#预测值
	predict=net(x)
	#预测值与真实值之间的误差，顺序不能颠倒
	loss=loss_func(predict,y)
	#所有参数的梯度降为0
	optimizer.zero_grad()
	#反向传播
	loss.backward()
	#优化梯度
	optimizer.step()


	#可视化训练过程
	if t%5==0:
		print(loss.data[0])
		plt.cla()
		plt.scatter(x.data.numpy(),y.data.numpy())
		plt.plot(x.data.numpy(),predict.data.numpy(),'r-',lw=5)
		plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color':  'red'})
		plt.pause(0.1)

