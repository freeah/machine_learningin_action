#GAN生成对抗网络
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np 
import matplotlib.pyplot as plt 

torch.manual_seed(1)
np.random.seed(1)

BATCH_SIZE=64
LR_G=0.0001  #generator生成器的学习率
LR_D=0.0001  #discriminator判别器的学习率

N_IDEAS=5
ART_COMPONENTS=15
PAINT_POINTS=np.vstack([np.linspace(-1,1,ART_COMPONENTS) for _ in range(BATCH_SIZE)]) 

#可视化图形
# plt.plot(PAINT_POINTS[0],2*np.power(PAINT_POINTS[0],2)+1,c='green',lw=3,label='line1')
# plt.plot(PAINT_POINTS[0],1*np.power(PAINT_POINTS[0],2)+0,c='red',lw=3,label='line2')
# plt.legend(loc='upper right')
# plt.show()

def artist_works():
	#从一个均匀分布[low,high)中随机采样，生成size个数据，注意定义域是左闭右开，即包含low，不包含high.
	#np.newaxis插入新的维度
	a=np.random.uniform(1,2,size=BATCH_SIZE)[:,np.newaxis]
	paintings=a*np.power(PAINT_POINTS,2)+(a-1)
	paintings=torch.from_numpy(paintings).float()
	return Variable(paintings)

G=nn.Sequential(
	nn.Linear(N_IDEAS,128), #输入随机想法，这是随机五个点
	nn.ReLU(),
	nn.Linear(128,ART_COMPONENTS) #生成的随机的点
	)	

D=nn.Sequential(
	nn.Linear(ART_COMPONENTS,128), #接收生成器生成的点
	nn.ReLU(),
	nn.Linear(128,1),  #输出判别真假
	nn.Sigmoid(), #百分比形式表示
	)

#优化器
opt_D=torch.optim.Adam(D.parameters(),lr=LR_D)
opt_G=torch.optim.Adam(G.parameters(),lr=LR_G)

#学习
for step in range(10000):
	artist_paintings=artist_works() #真实的
	G_ideas=Variable(torch.randn(BATCH_SIZE,N_IDEAS))
	G_paintings=G(G_ideas)  #生成器生成的

	#判别器进行鉴定
	prob_artist0=D(artist_paintings)
	prob_artist1=D(G_paintings)

	D_loss=-torch.mean(torch.log(prob_artist0)+torch.log(1-prob_artist1))  #GAN的误差公式
	G_loss=torch.mean(torch.log(1-prob_artist1))


	#反向传播
	opt_D.zero_grad()
	D_loss.backward(retain_graph=True) #保留一些参数
	opt_D.step()

	opt_G.zero_grad()
	G_loss.backward()
	opt_G.step()

	#可视化代码
	if step % 50 == 0:
		plt.cla()
		plt.plot(PAINT_POINTS[0], G_paintings.data.numpy()[0], c='#4AD631', lw=3, label='Generated painting',)
		plt.plot(PAINT_POINTS[0], 2 * np.power(PAINT_POINTS[0], 2) + 1, c='#74BCFF', lw=3, label='upper bound')
		plt.plot(PAINT_POINTS[0], 1 * np.power(PAINT_POINTS[0], 2) + 0, c='#FF9359', lw=3, label='lower bound')
		plt.text(-.5, 2.3, 'D accuracy=%.2f (0.5 for D to converge)' % prob_artist0.data.numpy().mean(), fontdict={'size': 13})
		plt.text(-.5, 2, 'D score= %.2f (-1.38 for G to converge)' % -D_loss.data.numpy(), fontdict={'size': 13})
		plt.ylim((0, 3));plt.legend(loc='upper right', fontsize=10);plt.draw();plt.pause(0.01)

plt.ioff()
plt.show()        


	
# artist_works()	