#定义一个网络
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

transform=transforms.Compose(
	[transforms.ToTensor(),  #数据标准化  (0,1)
	transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
	)

#第一个参数表示下载的文件放置在当前什么目录下
#第二个参数表示是训练集还是测试集，训练集为True   测试集为False
#第三个参数表示是否下载
#第四个参数表示原始数据需要转化成什么样的数据格式
trainset=torchvision.datasets.CIFAR10(root='./data',train=True,download=True,transform=transform)
#数据加载器
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,shuffle=True, num_workers=0)
testset=torchvision.datasets.CIFAR10(root='./data',train=False,download=True,transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,shuffle=False, num_workers=0)

classes=('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')

#图像可视化
def imshow(img):
    img = img / 2 + 0.5     # 非标准化
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# 得到一些随机的训练图像
dataiter = iter(trainloader)
images, labels = dataiter.next()

# 显示图像
imshow(torchvision.utils.make_grid(images))
# 输出类别
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

#定义一个卷积神经网络

class Net(nn.Module):
	def __init__(self):
		super(Net,self).__init__()
		#3表示输入图片的通道数，6表示输出通道数，5表示卷积核为5*5，第四个参数表示step 步长
		self.conv1=nn.Conv2d(3,6,5) 
		#2x2的filter 选取最大值
		self.pool=nn.MaxPool2d(2,2)
		self.conv2=nn.Conv2d(6,16,5)
		#仿射层/全连接层：y=wx+b
		self.fc1=nn.Linear(16*5*5,120)
		self.fc2=nn.Linear(120,84)
		self.fc3=nn.Linear(84,10)

	def forward(self,x):
		x=self.pool(F.relu(self.conv1(x))) # (3,32,32)->(6,28,28)->(6,28,28)->(6,14,14)
		x=self.pool(F.relu(self.conv2(x))) # (6,14,14)->(16,10,10)->(16,10,10)->(16,5,5)
		x=x.view(-1,16*5*5)
		x=F.relu(self.fc1(x))
		x=F.relu(self.fc2(x))
		x=self.fc3(x)
		return x

net=Net()
#使用交叉熵函数和随机梯度下降优化器
criterion=nn.CrossEntropyLoss()
optimizer=optim.SGD(net.parameters(),lr=0.003,momentum=0.9)
#训练网络
for epoch in range(2):
	running_loss=0.0
	for i,data in enumerate(trainloader, 0):
		inputs,labels=data
		inputs,labels=Variable(inputs),Variable(labels)
		optimizer.zero_grad()
		outputs=net(inputs)
		loss=criterion(outputs,labels)
		loss.backward()
		optimizer.step()

		running_loss+=loss.data[0]
		if i%2000==1999:
			print('[%d,%5d] loss:%.3f' %(epoch+1,i+1,running_loss/2000))
			running_loss=0.0

print('Finished Training')		

dataiter=iter(testloader)
images,labels=dataiter.next()
#打印图像
# imshow(torchvision.utils.make_grid(images))
# print('GroundTruth:',' '.join('%5s'%classes[labels[j]] for j in range(4)))	

#在测试集上执行
correct=0
total=0
for data in testloader:
	images,labels=data
	outputs=net(Variable(images))
	_,predicted=torch.max(outputs.data,1)
	total+=labels.size(0)
	correct+=(predicted==labels).sum()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))	


# print(net)	
#网络的可学习参数通过net.parameters()返回，net.named_parameters可同时返回学习的参数以及名称
# params=list(net.parameters())
# print(len(params))
# print(params[0].size())	#conv1的weight	

# input=Variable(torch.randn(1,1,32,32)) #网络的输入
# out=net(input) #网络的输出
# print(out)
# #损失函数
# target=Variable(torch.arange(1,11))
# criterion=nn.MSELoss() #在 nn 包下有几种不同的 损失函数 . 一个简单的损失函数是: nn.MSELoss 计算输出和目标之间的均方误差
# loss=criterion(out,target)
# print(loss)	

# #反向传播
# net.zero_grad()
# print('conv1.bias.grad before backward')
# print(net.conv1.bias.grad)

# loss.backward()

# print('conv1.bias.grad after backward')
# print(net.conv1.bias.grad)

# #更新权重
# learning_rate=0.01
# for f in net.parameters():
# 	f.data.sub_(f.grad.data*learning_rate)

#上述可以简写为一下
#新建一个优化器，指定要调整的参数和学习率
# optimizer=optim.SGD(net.parameters(),lr=0.01)
#在训练过程中
# optimizer.zero_grad()
# output=net(input)
# target=Variable(torch.arange(1,11))
# criterion=nn.MSELoss() #在 nn 包下有几种不同的 损失函数 . 一个简单的损失函数是: nn.MSELoss 计算输出和目标之间的均方误差
# loss=criterion(output,target)
# loss.backward()
# optimizer.step()
# print(loss)