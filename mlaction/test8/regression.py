import matplotlib
import matplotlib.pyplot as plt
from numpy import *
def readFile(filename):
	numFeat=len(open(filename).readline().split('\t'))-1
	fr=open(filename)
	dataMat=[];labelMat=[];
	for line in fr.readlines():
		lineArr=[]
		currLine=line.strip().split('\t')
		for i in range(numFeat):
			lineArr.append(float(currLine[i]))
		dataMat.append(lineArr)
		labelMat.append(float(currLine[-1]))
	return dataMat,labelMat	

xArr,yArr=readFile('ex0.txt')
#print(array(xArr)[:,1])
#分析数据
def drawPic():
	fig=plt.figure()
	ax1=fig.add_subplot(111)
	ax1.set_title('Scatter Plot')
	plt.xlabel('X')
	plt.ylabel('Y')
	ax1.scatter(array(xArr)[:,1],yArr)
	xCopy=mat(xArr).copy()
	yHat=xCopy*ws #画回归线
	ax1.plot(xCopy[:,1],yHat)
	plt.show()
#drawPic()

#线性回归求w
def standRegres(xArr,yArr):
	xMat=mat(xArr);yMat=mat(yArr).T
	xTx=xMat.T*xMat
	#linalg.det() 矩阵求行列式，如果行列式等于0，则不可逆
	if linalg.det(xTx)==0:
		print('this matrix is singular,can not do inverse')
		return
	#.I求矩阵的逆	
	ws=xTx.I*xMat.T*yMat  #根据推出来的公式计算
	return ws #返回回归系数
ws=standRegres(xArr,yArr)
#drawPic()	

#计算预测值和真实值的匹配程度，求两个序列的相关系数，保证两个向量都是行向量
xMat=mat(xArr);yMat=mat(yArr)
yHat=xMat*ws
#print(corrcoef(yHat.T,yMat))

#局部加权线性回归
def lwlrTest(testArr,xArr,yArr,k=1.0):
	m=shape(testArr)[0]
	yHat=zeros(m)
	for i in range(m):
		yHat[i]=lwlr(testArr[i],xArr,yArr,k)
	return yHat
def lwlr(testPoint,xArr,yArr,k=1.0):
	xMat=mat(xArr);yMat=mat(yArr).T
	m=shape(xMat)[0]
	weights=mat(eye((m)))
	for j in range(m):
		diffMat=testPoint-xMat[j,:]
		weights[j,j]=exp(diffMat*diffMat.T/(-2.0*k**2))
	xTx=xMat.T*(weights*xMat)
	if linalg.det(xTx)==0.0:
		print('this matrix is singular can not do inverse')
		return
	ws=xTx.I*(xMat.T*(weights*yMat))
	return testPoint*ws		

#可以比较不同的k值出来的结果
yHat=lwlrTest(xArr,xArr,yArr,0.003)	

#分析误差的大小可以用loss function来表示
def rssError(yArr,yHatArr):
	return ((yArr-yHatArr)**2).sum()
print(rssError(yArr,yHat))

#岭回归，当特征数量>数据的个数
def ridgeRegres(xMat,yMat,lam=0.2):
	xTx=xMat.T*xMat
	denom=xTx+eye(shape(xMat)[1])*lam
	if linalg.det(denom)==0.0:
		print('this matrix is singular,can not do inverse')
		return
	ws=denom.I*(xMat.T*yMat)
	return ws

def ridgeTest(xArr,yArr):
	xMat=mat(xArr);yMat=mat(yArr).T
	print('xMat',xMat);print('yMat',yMat)
	#对数据进行标准化，使每维特征具有相同的重要性。具体的做法是所有特征都减去各自的均值并除以方差
	#计算每列的均值
	yMean=mean(yMat,0)
	yMat=yMat-yMean
	print('yMean',yMean);print('yMat',yMat)
	xMeans=mean(xMat,0)
	#计算每列的方差
	xVar=var(xMat,0)
	xMat=(xMat-xMeans)/xVar
	numTestPts=30
	wMat=zeros((numTestPts,shape(xMat)[1]))
	print('xMat',xMat)
	#对不同的 lam 计算 ws 回归系数
	for i in range(numTestPts):
		ws=ridgeRegres(xMat,yMat,exp(i-10))
		wMat[i,:]=ws.T
	return wMat
				
abX,abY=readFile('abalone.txt')
#ridgeWeights=ridgeTest(abX,abY)
#用缩减法确定最佳回归系数，交叉验证测试岭回归
#步骤
#90%的训练数据  10%的测试数据
#使用训练数据求出权重，对应于不同lam的权重向量，构成一个矩阵
#使用测试数据对每一行权重向量求出对应的y向量，
#求出预测的y和实际的y的误差，找到最小的误差，就找到使误差最小的权重向量
import random
def crossValidation(xArr,yArr,numVal=10):
	m=len(yArr) #4177行
	indexList=list(range(m))
	errorMat=zeros((numVal,30))
	for i in range(numVal):
		trainX=[];trainY=[]
		testX=[];testY=[]
		random.shuffle(indexList)
		for j in range(m):
			if j<int(m*0.9):
				trainX.append(xArr[indexList[j]])  #训练集
				trainY.append(yArr[indexList[j]])
			else:
				testX.append(xArr[indexList[j]])  #测试集
				testY.append(yArr[indexList[j]])
	wMat=ridgeTest(mat(trainX),mat(trainY))
	print('wMat',wMat);print(shape(wMat))  #30行8列
	for k in range(30):
		matTestX=mat(testX);matTrainX=mat(trainX)
		meanTrain=mean(matTrainX,0)
		varTrain=var(matTrainX,0)
		matTestX=(matTestX-meanTrain)/varTrain
		yEst=matTestX*mat(wMat[k,:]).T+mean(trainY) #对于标准化数据还是模糊
		errorMat[i,k]=rssError(yEst.T.A,mat(testY).A) #每个lam值对应的误差
	meanErrors=mean(errorMat,0)
	print('meanErrors',meanErrors)
	minMean=float(min(meanErrors))
	print('minMean',minMean)
	#nonzeros(a)返回数组a中值不为零的元素的下标，它的返回值是一个长度为a.ndim(数组a的轴数)的元组，
	#元组的每个元素都是一个整数数组，其值为非零元素的下标在对应轴上的值。
	print(nonzero(meanErrors==minMean))  #14
	bestWeights=wMat[nonzero(meanErrors==minMean)]
	xMat=mat(xArr);yMat=mat(yArr).T
	meanX=mean(xMat,0);varX=var(xMat,0)
	unReg=bestWeights/varX
	print('the best model from Ridge Regression is :\n',unReg)
	print('with constant term:',-1*sum(multiply(meanX,unReg))+mean(yMat))				

crossValidation(abX,abY,10)

# fig=plt.figure()
# ax=fig.add_subplot(111)
# ax.plot(ridgeWeights)
# plt.show()

#前向逐步线性回归
def stageWise(xArr,yArr,eps=0.01,numIt=100):
	xMat=mat(xArr);yMat=mat(yArr).T
	#标准化数据
	yMean=mean(yMat,0)
	yMat=yMat-yMean
	xMat=(xMat-mean(xMat,0))/var(xMat,0)
	m,n=shape(xMat)
	returnMat=zeros((numIt,n))
	ws=zeros((n,1));weTest=ws.copy();wsMax=ws.copy()
	#迭代
	for i in range(numIt):
		print(ws.T)
		lowestError=inf
		#每个特征循环
		for j in range(n):
			for sign in [-1,1]:
				wsTest=ws.copy()
				wsTest[j]+=eps*sign
				yTest=xMat*wsTest
				rssE=rssError(yMat.A,yTest.A)
				if rssE<lowestError:
					lowestError=rssE
					wsMax=wsTest
		ws=wsMax
		returnMat[i,:]=ws.T
	return returnMat			

#print(stageWise(abX,abY,0.01,200))	

#总结
# 与分类一样，回归也是预测目标值的过程。回归与分类的不同点在于，前者有预测连续型变量，而后者预测离散型变量。
# 回归是统计学中最有力的工具之一。在回归方程里，球的特征对应的最佳回归系数的方法是最小化误差的平方和。
# 给定输入矩阵X，如果X.T*X的逆矩阵存在并可以求得的话，回归法都可以直接使用。数据集上计算出的回归方程
# 并不一定意味着他是最佳的，可以使用预测值y和原始值y的相关性来度量回归方程的好坏。

# 当数据的样本数比特征数还少的时候，矩阵X.T*X的逆不能直接计算。这时可以考虑岭回归。

# 岭回归是缩减法的一种，相当于对回归系数的大小施加了限制。另一种很好的缩减法是lasso，Lasso
# 难以求解，但可以使用计算简单的逐步线性回归方法来求得近似结果。

# 缩减法还可以看作是对一个模型增加偏差的同时减少方差。偏差方差折中是一个重要的概念，
# 可以帮助我们理解现有模型并作出改进，从而得到更好的模型。