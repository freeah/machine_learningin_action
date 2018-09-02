import matplotlib
import matplotlib.pyplot as plt
from numpy import *
#分析数据
def readData(filename):
	fr=open(filename)
	#dataNum=len(fr.readline().strip().split())
	dataSet=[]
	for line in fr.readlines():
		currLine=line.strip().split()
		lineArr=[]
		for i in range(16):
			lineArr.append(float(currLine[i]))
		dataSet.append(lineArr)
	return dataSet

xArr=mat(readData('1_test.txt')).T.getA()
yArr=readData('1_test_targets.txt')

def drawPic():
	fig=plt.figure()
	ax1=fig.add_subplot(111)
	ax1.set_title('Scatter Plot')
	plt.xlabel('X')
	plt.ylabel('Y')
	ax1.plot(xArr)
	plt.show()
drawPic()


#局部加权线性回归
def lwlrTest(testArr,xArr,yArr,k=1.0):
	m=shape(testArr)[0]
	yHat=zeros(m)
	for i in range(m):
		yHat[i]=lwlr(testArr[i],xArr,yArr,k)
	return yHat
def lwlr(testPoint,xArr,yArr,k=1.0):
	print(testPoint)
	xMat=mat(xArr);yMat=mat(yArr).T
	m=shape(xMat)[0]
	weights=mat(eye((m)))
	print(weights)
	print(xMat[0,:])
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
print(yHat)		

#分析误差的大小可以用loss function来表示
def rssError(yArr,yHatArr):
	return ((yArr-yHatArr)**2).sum()
print(rssError(yArr,yHat))