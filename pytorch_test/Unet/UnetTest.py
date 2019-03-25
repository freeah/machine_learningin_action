import torch.nn as nn
import torch
from torch import autograd
from torchvision.datasets import ImageFolder
from torchvision import transforms,utils
from torch.utils.data import DataLoader,TensorDataset
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import PIL.Image as Image
import os
#数据

transform=transforms.Compose([
	transforms.ToTensor(),
	#transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
def make_dataset(root):
    imgs=[]
    n=len(os.listdir(root))
    rootdir=os.listdir(root)
    print(rootdir)
    for j in range(30):
    	if rootdir[0]=="image":
    		img_path=os.path.join(root,rootdir[0]+'/')
    		label_path=os.path.join(root,rootdir[1]+'/')
    	if rootdir[0]=="label":
    		img_path=os.path.join(root,rootdir[1]+'/')
    		label_path=os.path.join(root,rootdir[0]+'/')
    	img=os.path.join(img_path,'%d.png'%j)
    	img=transform(Image.open(img))
    	label=os.path.join(label_path,'%d.png'%j)
    	label=transform(Image.open(label))
    	imgs.append((img,label))   	
    return imgs	

def make_testdataset(root):
	imgs=[]
	testimg_path=os.listdir(root)
	m=len(testimg_path)
	for i in range(m):
		img_tpath=os.path.join(root,testimg_path[i])
		img_test=transform(Image.open(img_tpath))
		imgs.append((img_test,'_'))
	return imgs	


liver_dataset = make_dataset("data/membrane/train/")
dataloaders = DataLoader(liver_dataset, batch_size=2,shuffle=True)

test_dataset=make_testdataset('data/membrane/test/')
testloaders = DataLoader(test_dataset, batch_size=1,shuffle=False)

#是否使用cuda
print(torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

#模型
class DoubleConv(nn.Module):
	def __init__(self,in_ch,out_ch):
		super(DoubleConv,self).__init__()
		self.conv=nn.Sequential(
			nn.Conv2d(in_ch,out_ch,3,padding=1),
			nn.BatchNorm2d(out_ch),
			nn.ReLU(),
			nn.Conv2d(out_ch,out_ch,3,padding=1),
			nn.BatchNorm2d(out_ch),
			nn.ReLU()
			)

	def forward(self,input):
		return self.conv(input)	


class Unet(nn.Module):
	def __init__(self,in_ch,out_ch):
		super(Unet,self).__init__()
		#下采样过程
		self.conv1=DoubleConv(in_ch,64)
		self.pool1=nn.MaxPool2d(2)
		self.conv2=DoubleConv(64,128)
		self.pool2=nn.MaxPool2d(2)
		self.conv3=DoubleConv(128,256)
		self.pool3=nn.MaxPool2d(2)
		self.conv4=DoubleConv(256,512)
		self.pool4=nn.MaxPool2d(2)
		self.conv5=DoubleConv(512,1024)
		#上采样过程
		self.up1=nn.ConvTranspose2d(1024,512,2,stride=2)
		self.conv6=DoubleConv(1024,512)
		self.up2=nn.ConvTranspose2d(512,256,2,stride=2)
		self.conv7=DoubleConv(512,256)
		self.up3=nn.ConvTranspose2d(256,128,2,stride=2)
		self.conv8=DoubleConv(256,128)
		self.up4=nn.ConvTranspose2d(128,64,2,stride=2)
		self.conv9=DoubleConv(128,64)
		self.conv10=nn.Conv2d(64,out_ch,1)

	def forward(self,x):
		#下采样
		c1=self.conv1(x)
		#print('c1.size()',c1.size())
		p1=self.pool1(c1)
		#print('p1.size()',p1.size())	
		c2=self.conv2(p1)
		#print('c2.size()',c2.size())
		p2=self.pool2(c2)
		#print('p2.size()',p2.size())
		c3=self.conv3(p2)
		#print('c3.size()',c3.size())
		p3=self.pool3(c3)
		#print('p3.size()',p3.size())
		c4=self.conv4(p3)
		#print('c4.size()',c4.size())
		p4=self.pool4(c4)
		#print('p4.size()',p4.size())
		c5=self.conv5(p4)
		#print('c5.size()',c5.size())
		#上采样
		u1=self.up1(c5)
		#print('u1.size()',u1.size())
		merge1=torch.cat([u1,c4],dim=1)
		#print('merge1.size()',merge1.size())
		c6=self.conv6(merge1)
		#print('c6.size()',c6.size())
		u2=self.up2(c6)
		#print('u2.size()',c6.size())
		merge2=torch.cat([u2,c3],dim=1)
		#print('merge2.size()',c6.size())
		c7=self.conv7(merge2)
		#print('c7.size()',c7.size())
		u3=self.up3(c7)
		#print('u3.size()',u3.size())
		merge3=torch.cat([u3,c2],dim=1)
		#print('merge3.size()',merge3.size())
		c8=self.conv8(merge3)
		#print('c8.size()',c8.size())
		u4=self.up4(c8)
		#print('u4.size()',u4.size())
		merge4=torch.cat([u4,c1],dim=1)
		#print('merge4.size()',merge4.size())
		c9=self.conv9(merge4)
		#print('c9.size()',c9.size())
		c10=self.conv10(c9)
		#print('c10.size()',c10.size())
		out = nn.Sigmoid()(c10)
		#print('out.size()',out.size())

		return out

unet=Unet(1,1).to(device)	
# x = Variable(torch.FloatTensor(np.random.random((2, 512, 512))))
# y=unet(x)

#训练
optimizer=torch.optim.Adam(unet.parameters(),lr=3e-5)
#loss_fun=nn.CrossEntropyLoss()
loss_fun=nn.BCELoss()
epochs=300
for epoch in range(epochs):
	epoch_loss=0
	for i,(train_x,train_y) in enumerate(dataloaders):
		inputs=train_x.to(device)
		labels=train_y.to(device)
		out_put=unet(inputs)
		loss=loss_fun(out_put,labels)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		epoch_loss+=loss.item()
		#print(i+1,'|',epoch + 1, 'loss',loss.item())
	print('epoch',epoch,'| loss:%.3f' % epoch_loss)

torch.save(unet.state_dict(),'weights_%d.pth'% epoch)

print('finish training.....')	

print('test starting')

#测试
for i,(test_x,test_y) in enumerate(testloaders):
	test_inputs=test_x.to(device)
	predict=unet(test_inputs).cpu().clone()
	print(predict.type(),predict.size())
	predict=torch.squeeze(predict,0)
	print(predict.type(),predict.size())
	predict=transforms.ToPILImage()(predict) #Tensor转化为PIL图片保存
	predict.save('results/%s_predict.png' % i)

print('test endding')	

