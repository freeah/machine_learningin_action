import numpy as np 
import random
import os,struct
from array import array as pyarray
from numpy import append,array,int8,uint8,zeros
import matplotlib.pyplot as plt

class NeuralNet(object):
	def __init__(self,sizes):
		self.sizes_=sizes
		self.num_layers_=len(sizes) #设置网络的层数
		#np.random.randn函数生成平均值为0且标准差为1的高斯分布
		self.w_=[np.random.randn(y,x) for x,y in zip(sizes[:-1],sizes[1:])] #设置权重
		self.b_=[np.random.randn(y,1) for y in sizes[1:]]  #设置偏置

	#定义sigmoid函数，作为激活函数
	def sigmoid(self,z):
		return 1.0/(1.0+np.exp(-z))

	#sigmoid函数的导函数 上面的公式推导出来
	def sigmoid_prime(self,z):
		return self.sigmoid(z)*(1-self.sigmoid(z))	

	#定义feedforward函数，前向传播算法

	# 假设上一层结点i,j,k,…等一些结点与本层的结点w有连接，那么结点w的值怎么算呢？
	# 就是通过上一层的i,j,k等结点以及对应的连接权值进行加权和运算，
	# 最终结果再加上一个偏置项
	# 最后在通过一个非线性函数（即激活函数），
	# 如ReLu，sigmoid等函数，最后得到的结果就是本层结点w的输出。 
	def feedforward(self,x):
		for b,w in zip(self.b_,self.w_):
			x=self.sigmoid(np.dot(w,x)+b) #np.dot()向量点积
		return x	

	#反向传播算法（Back propagation）
	#https://blog.csdn.net/bitcarmanlee/article/details/78819025
	#http://neuralnetworksanddeeplearning.com/chap1.html
	#http://neuralnetworksanddeeplearning.com/chap2.html

	#training_data是训练数据，epochs是训练次数是13，mini_batch_size是每次训练样本数是100，eta是学习率3.0
	def SGD(self,training_data,epochs,mini_batch_size,eta,test_data=None):
		if test_data:
			n_test=len(test_data)

		n=len(training_data)
		
		for j in range(epochs):
			random.shuffle(training_data) #随机排序
			mini_batches=[training_data[k:k+mini_batch_size] for k in range(0,n,mini_batch_size)] #100个数据为一组构成的数组，每次训练的样本数是100
			for mini_batch in mini_batches:
				self.update_mini_batch(mini_batch,eta)  #训练，调整w  b

			#计算预测值和实际值相等的个数/总测试集的个数	
			if test_data:
				print('Epoch {0}:{1}/{2}'.format(j,self.evaluate(test_data),n_test))
			else:
				print('Epoch {0} complete'.format(j))

	def backprop(self,x,y):
		nabla_b=[np.zeros(b.shape) for b in self.b_]
		nabla_w=[np.zeros(w.shape) for w in self.w_]

		activation=x
		activations=[x]
		zs=[]
		for b,w in zip(self.b_,self.w_):
			z=np.dot(w,activation)+b  #wx+b
			zs.append(z)
			activation=self.sigmoid(z) #使用激活函数
			activations.append(activation)

		delta=self.cost_derivative(activations[-1],y)*self.sigmoid_prime(zs[-1]) #公式
		#根据公式求第二层到第三层即最后一层的，误差对w和b的偏导数
		nabla_b[-1]=delta
		nabla_w[-1]=np.dot(delta,activations[-2].transpose())
		#根据公式，误差对每一层的w  b的偏导数
		for i in range(2, self.num_layers_):
			z = zs[-i]
			sp = self.sigmoid_prime(z)
			delta = np.dot(self.w_[-i+1].transpose(), delta) * sp
			nabla_b[-i] = delta
			nabla_w[-i] = np.dot(delta, activations[-i-1].transpose())
		return (nabla_b,nabla_w)
	#调整w  b
	def update_mini_batch(self,mini_batch,eta):
		nabla_b=[np.zeros(b.shape) for b in self.b_]
		nabla_w=[np.zeros(w.shape) for w in self.w_]
		for x,y in mini_batch:
			delta_nabla_b,delta_nabla_w=self.backprop(x,y)  #反向传播
			nabla_b=[nb+dnb for nb,dnb in zip(nabla_b,delta_nabla_b)]
			nabla_w=[nw+dnw for nw,dnw in zip(nabla_w,delta_nabla_w)]
		self.w_ = [w-(eta/len(mini_batch))*nw for w, nw in zip(self.w_, nabla_w)]
		self.b_ = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.b_, nabla_b)]

	#计算预测值和实际值相等的个数，对测试数据的X计算，输出预测的Y，预测的Y与实际的Y进行比较  如果相等  即int(x == y)=1
	def evaluate(self,test_data):
		test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]  #argmax返回的是最大数的索引
		return sum(int(x == y) for (x, y) in test_results)

	def cost_derivative(self,output_activations,y):
		return (output_activations-y)

	#预测
	def predict(self,data):
		value=self.feedforward(data)
		return value.tolist().index(max(value))	#tolist()将数组或者矩阵转换成列表 

	# 保存训练模型
	def save(self):
		pass

	def load(self):
		pass	

#加载MNIST数据集
def load_mnist(dataset="training_data",digits=np.arange(10),path="."):
	if dataset=="training_data":
		fname_image=os.path.join(path,"train-images-idx3-ubyte") #定义文件路径
		fname_label=os.path.join(path,"train-labels-idx1-ubyte")
		print(fname_image,fname_label)
	elif dataset=="testing_data":
		fname_image=os.path.join(path,"t10k-images-idx3-ubyte")	
		fname_label = os.path.join(path, 't10k-labels-idx1-ubyte')
	else:
		raise ValueError("dataset must be 'training_data' or 'testing_data'")

	flbl = open(fname_label, 'rb') #以二进制格式打开一个文件用于只读。文件指针将会放在文件的开头。这是默认模式。一般用于非文本文件如图片等。

	#读取文件内容
	magic_nr, size = struct.unpack(">II", flbl.read(8))
	lbl = pyarray("b", flbl.read())
	flbl.close()

	fimg = open(fname_image, 'rb')
	magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
	img = pyarray("B", fimg.read())
	fimg.close()

	print(magic_nr,size,rows,cols,digits)

	ind = [ k for k in range(size) if lbl[k] in digits ]
	N = len(ind)

	print(N)

	images = zeros((N, rows, cols), dtype=uint8) #定义三维数组，存放数据集数组
	labels = zeros((N, 1), dtype=int8) #定义标签，存放数据集标签
	for i in range(len(ind)):
		images[i] = array(img[ind[i]*rows*cols : (ind[i]+1)*rows*cols ]).reshape((rows, cols))
		labels[i] = lbl[ind[i]]

	return images, labels
    
def load_samples(dataset="training_data"):
    image,label = load_mnist(dataset)  #读出的图像和对应的数字, 
    print(image)
    print(label)
    #画图
    # fig,ax=plt.subplots(nrows=2,ncols=5,sharex=True,sharey=True)
    # ax=ax.flatten()
    # for i in range(10):
    # 	# img=image[label==i][0].reshape(28,28)
    # 	ax[i].imshow(image[i],cmap="Greys",interpolation="nearest")

    # ax[0].set_xticks([])
    # ax[0].set_yticks([])
    # plt.tight_layout()
    # plt.show()	




    #print(image[0].shape, image.shape)   # (28, 28) (60000, 28, 28)
    #print(label[0].shape, label.shape)   # (1,) (60000, 1)
    #print(label[0])   # 5

    # a=array([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]])
    # A=[np.reshape(i,(6,1)) for i in a]
    # print(A)
    # print([i/20.0 for i in A])

    # 每张图片是28*28的像素，把28*28二维数据转为一维数据
    X = [np.reshape(x,(28*28, 1)) for x in image]

    X = [x/255.0 for x in X]   # 灰度值范围(0-255)，转换为(0-1)  数组里面的元素代表每个图像构成的28*28行一列的数据

    # 5 -> [0,0,0,0,0,1.0,0,0,0]      1 -> [0,1.0,0,0,0,0,0,0,0]
    def vectorized_Y(y):
        e = np.zeros((10, 1))
        e[y] = 1.0
        return e
    # 把Y值转换为神经网络的输出格式
    if dataset == "training_data":
        Y = [vectorized_Y(y) for y in label]
        pair = list(zip(X, Y)) #将X和对应的Y打包成一个个元组，然后返回由这些元组组成的对象
        return pair
    elif dataset == 'testing_data':
        pair = list(zip(X, label))
        return pair
    else:
        print('Something wrong') 		

if __name__ == '__main__':
    INPUT = 28*28
    OUTPUT = 10

    # net=NeuralNet([3,4,2])
    # print('权重：',net.w_)
    # print('偏置：',net.b_)
    #定义一个网络，三层
    net = NeuralNet([INPUT, 40, OUTPUT])

 	#读取数据，返回的是数组，数组中的元素是一个个元组，元组的元素是X和对应的Y
    train_set = load_samples(dataset='training_data')
    test_set = load_samples(dataset='testing_data')

 	#训练数据
    net.SGD(train_set, 13, 100, 2.0, test_data=test_set)
 
    #准确率
    correct = 0;
    for test_feature in test_set:
        if net.predict(test_feature[0]) == test_feature[1][0]:
            correct += 1
    print("准确率: ", correct/len(test_set))        


# net=NeuralNet([3,4,2])
# print('权重：',net.w_)
# print('偏置：',net.b_)
# print(list(zip(net.b_,net.w_)))

#print([3,4,2][:-1]) #[3,4]
#print([3,4,2][1:]) #[4,2]
#for x,y in zip([3,4,2][:-1],[3,4,2][1:]):
	#print(x,y)

