#关系拟合(回归)
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

n_data=torch.ones(100,2)
x0=torch.normal(2*n_data,1)
y0=torch.zeros(100) #一类是0
x1=torch.normal(-2*n_data,1)
y1=torch.ones(100) #一类是1

x=torch.cat((x0,x1),0).type(torch.FloatTensor)
y=torch.cat((y0,y1),0).type(torch.LongTensor)  #标签一定是这种形式
#x，y转化成Variable形似，神经网络只能输入Variable
x,y=Variable(x),Variable(y)
# plt.scatter(x.data.numpy()[:,0],x.data.numpy()[:,1],c=y.data.numpy(),s=100,lw=0,cmap='RdYlGn')
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

net=Net(2,10,2)  #如果分类输出为[0,1]则分为1类  若为[1,0]则为0类
print(net)		

#优化器优化神经网络参数,传入所有参数
optimizer=torch.optim.SGD(net.parameters(),lr=0.02)
#loss函数，主要针对分类问题，softmax   概率
loss_func=torch.nn.CrossEntropyLoss()

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
	if t%2==0:
		print(loss.data[0])
		plt.cla()
		#过了一道softmax的激励函数后的最大概率才是预测值
		prediction=torch.max(F.softmax(predict),1)[1]
		pred_y=prediction.data.numpy().squeeze()
		target_y=y.data.numpy()
		plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
		accuracy = sum(pred_y == target_y)/200.  # 预测中有多少和真实值一样
		plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color':  'red'})
		plt.pause(0.1)

plt.ioff()  # 停止画图
plt.show()        

