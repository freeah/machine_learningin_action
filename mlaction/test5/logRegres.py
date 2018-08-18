#logistic回归分类器
#在每个特征上乘以一个回归系数，然后把所有的值相加，将这个总和带入sigmoid函数中，进而得到一个范围在0-1之间的数值
#任何大于0.5的数据被分入1类，小于0.5的即被归入0类，所以logistic回归也可以看做是一种概率估计
from numpy import *
#Logistic回归梯度上升优化算法
def loadDataSet():
	dataMat=[];labelMat=[]
	fr=open('testSet.txt')
	for line in fr.readlines():
		lineArr=line.strip().split()
		dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])]) #为了方便计算，x0设置为1，后面分别是x1，x2
		labelMat.append(int(lineArr[2]))  #文件中第三列，类别标签
	return dataMat,labelMat	

#sigmoid函数
def sigmoid(inX):
	return 1.0/(1+exp(-inX))
#梯度上升法
def gradAsent(dataMatIn,classLabels):
	dataMatrix=mat(dataMatIn)  #.mat()将数据转化为矩阵，进行矩阵操作
	labelMat=mat(classLabels).transpose() #转置
	m,n=shape(dataMatrix) #m行n列
	alpha=0.001
	maxCycles=500  #迭代次数
	weights=ones((n,1)) #每个回归系数初始设为1
	for k in range(maxCycles):
		#z=wx z就是sigmoid函数输入值，得到一个在0-1之间的数值，<0.5是一类  >0.5是一类
		h=sigmoid(dataMatrix*weights) 
		#梯度求解，可以看李宏毅http://speech.ee.ntu.edu.tw/~tlkagk/courses/ML_2017/Lecture/Regression.pdf这节
		#https://blog.csdn.net/kaka19880812/article/details/46993917也有解释
		error=(labelMat-h)
		weights=weights+alpha*dataMatrix.transpose()*error  #更新回归系数
	return weights		

#dataArr,labelMat=loadDataSet()		
#print(gradAsent(dataArr,labelMat))

#分析数据：画出决策边界
def plotBestFit(wei):
	import matplotlib.pyplot as plt
	weights=wei
	dataMat,labelMat=loadDataSet()
	dataArr=array(dataMat)
	n=shape(dataArr)[0]
	print(n)
	xcord1=[];ycord1=[]  #类别为1的x,y值
	xcord2=[];ycord2=[]  #类别为0的x,y值
	for i in range(n):
		if int(labelMat[i])==1:
			xcord1.append(dataArr[i,1]);ycord1.append(dataArr[i,2])
		else:
			xcord2.append(dataArr[i,1]);ycord2.append(dataArr[i,2])
	fig=plt.figure()
	ax=fig.add_subplot(111)
	#散点图，两个类别两个颜色
	ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
	ax.scatter(xcord2,ycord2,s=30,c='green')
	#start end step
	x=arange(-3.0,3.0,0.1)
	#sigmoid()函数  >0.5是一类  <0.5是一类  此时z=0  
	#所以0=w0x0+w1x1+w2x2  x0=1  x1和x2的关系可以如下表示
	y=(-weights[0]-weights[1]*x)/weights[2]
	ax.plot(x,y)
	plt.xlabel('X1');plt.ylabel('X2')
	plt.show()		

dataArr,labelMat=loadDataSet()		
weights=gradAsent(dataArr,labelMat)
#Return self as an ndarray object.
#weihts是矩阵形式，不能取出每一行每一列的值，转换成ndarray对象，就可以取出了
#plotBestFit(weights.getA())		

#梯度上升算法在每次更新回归系数时都需要遍历整个数据集，该方法在处理100个左右的数据集时尚可，但如果有数十亿样本
#和成千上万的特征，那么该方法的计算复杂度就太高了。一种改进的方法是一次仅用一个样本点来更新回归系数
#该方法称为随机梯度上升法

#改进的随机梯度上升算法
def stocGradAsent1(dataMatrix,classLabels,numIter=150):
	m,n=shape(dataMatrix)
	weights=ones(n)
	for j in range(numIter):
		dataIndex=list(range(m))
		for i in range(m):
			alpha=4/(1.0+j+i)+0.01
			randIndex=int(random.uniform(0,len(dataIndex)))
			h=sigmoid(sum(dataMatrix[randIndex]*weights))
			error=classLabels[randIndex]-h
			weights=weights+alpha*error*dataMatrix[randIndex]
			del(dataIndex[randIndex])
	return weights		

# dataArr,labelMat=loadDataSet()		
# weights=stocGradAsent1(array(dataArr),labelMat)
# plotBestFit(weights)		

#从疝气病症预测病马的死亡率

#收集数据：给定数据文件
#准备数据：用Python解析文本文件并填充缺失值。
#分析数据：可视化并观察数据
#训练算法：使用优化算法，找到最佳系数
#测试算法：为了量化回归的效果，需要观察错误率。根据错误率决定是否回退到训练阶段，通过改变迭代次数和步长等参数来得到更好的回归系数。
#使用算法
def classifiVector(intX,weights):
	prob=sigmoid(sum(intX*weights))
	if prob>0.5:
		return 1.0
	else:
		return 0.0	

#使用一个文件数据进行训练，选练出权重
#使用另一个文件进行测试，比较预测出的类别与实际类别比较，计算错误率
def colicTest():
	frTrain=open('horseColicTraining.txt')	
	frTest=open('horseColicTest.txt')
	trainingSet=[];trainingLbels=[]
	for line in frTrain.readlines():
		currLine=line.strip().split('\t')
		lineArr=[]
		for i in range(21):
			lineArr.append(float(currLine[i]))
		trainingSet.append(lineArr)	
		trainingLbels.append(float(currLine[21]))
	trainWeights=stocGradAsent1(array(trainingSet),trainingLbels,500)	#得到权重
	errorCount=0;numTestVec=0.0
	#测试的数据
	for line in frTest.readlines():
		numTestVec+=1.0
		currLine=line.strip().split('\t')
		lineArr=[]
		for i in range(21):
			lineArr.append(float(currLine[i]))
		#带入函数，测试计算出的类别与实际类别相符性	
		if int(classifiVector(array(lineArr),trainWeights))!=int(currLine[21]):
			errorCount+=1
	errorRate=(float(errorCount)/numTestVec)
	print('the error rate of this test is:%f'%errorRate)
	return errorRate	

colicTest()	

def multiTest():
	numTests=10;errorSum=0.0
	for k in range(numTests):
		errorSum+=colicTest()
	print('after %d iterations the average error rate is:%f'%(numTests,errorSum/float(numTests)))	

multiTest()	