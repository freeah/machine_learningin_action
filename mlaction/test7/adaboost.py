#Adaboost迭代算法步骤：
#1.初始化训练数据的权值分布。如果有N个样本，则每一个训练样本最开始时都被赋予相同的权值：1/N。
#2.训练弱分类器。具体训练过程中，如果某个样本点已经被准确的分类，那么在构造下一个训练集中，他的权值就被降低；
#相反，如果某个样本点没有被准确地分类，那么他的权值就会提高。然后，权值更新过的样本集被用于训练下一个分类器，整个训练过程如此迭代地进行下去。
#3.将各个训练得到的弱分类器组合成强分类器。各个弱分类器的训练过程结束后，加大分类误差率小的弱分类器的权重，
#使其在最终的分类函数中起着较大的决定作用，而降低误差率大啊的弱分类器的权重，使其最终的分类函数中起着较小
#的决定作用。换言之，误差率低的弱分类器在最终的分类器中占得权重较大，否则较小

#运行过程如下：
# 训练数据中的每个样本，并赋予其权重，这些权重构成了向量D，一开始这些权重都初始化成相等值，
# 首先在训练器上训练出一个弱分类器并计算该分类器的错误率，然后在同一数据集上在此训练弱分类器
# 在分类器的第二次训练当中，将会重新调整每个样本的权重，其中第一次分对的样本的权重会降低，而第一次
# 分错的样本的权重会提高，AdaBoost为每个分类器都分配了一个权重值alpha，这些alpha值是基于每个弱分类器
# 的错误率进行计算的，
# 首先计算错误率，即误分类样本的权值之和
# 之后计算alpha值：0.5*ln((1-错误率)/错误率)
# 更新权重
# 进行下一轮迭代，直到训练错误率为0或者弱分类器的数目达到用户的指定值为止

#参考 https://blog.csdn.net/v_july_v/article/details/40718799
from numpy import *
def loadSimpData():
	dataMat=matrix([
		[1.,2.1],
		[2.,1.1],
		[1.3,1.],
		[1.,1.],
		[2.,1.]
		])
	classLabels=[1.0,1.0,-1.0,-1.0,1.0]
	return dataMat,classLabels	

#单层决策树生成函数
def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):
	retArray=ones((shape(dataMatrix)[0],1))
	if threshIneq=='It':
		#dataMatrix中满足dataMatrix[:,dimen]<=threshVal的位置，将retArray与之相对应的位置的值变为-1
		retArray[dataMatrix[:,dimen]<=threshVal]=-1.0 
	else:
		retArray[dataMatrix[:,dimen]>threshVal]=-1.0
	return retArray

def buildStump(dataArr,classLabels,D):
	dataMatrix=mat(dataArr);labelMat=mat(classLabels).T	
	#print('dataMatrix',dataMatrix)
	#print('labelMat',labelMat)
	m,n=shape(dataMatrix) #5,2
	numSteps=10.0;bestStump={};bestClasEst=mat(zeros((m,1)))
	minError=inf
	#遍历每个特征
	for i in range(n):
		#通过计算最小值和最大值来了解应该需要多大的步长
		rangeMin=dataMatrix[:,i].min();rangeMax=dataMatrix[:,i].max();
		stepSize=(rangeMax-rangeMin)/numSteps
		#在值上遍历
		for j in range(-1,int(numSteps)+1):
			for inequal in ['It','gt']:
				threshVal=(rangeMin+float(j)*stepSize) #设定阙值
				predictedVals=stumpClassify(dataMatrix,i,threshVal,inequal) #进行分类预测
				errArr=mat(ones((m,1)))
				#print('predictedVals',predictedVals)
				#print('labelMat',labelMat)
				errArr[predictedVals==labelMat]=0 #预测与label相等则为0，否则为1
				#print('errArr',errArr)
				weightedError=D.T*errArr #计算误差率 即误分类样本的权值之和
				#print('split:dim %d,thresh %.2f,thresh inequal:%s,the weighted error is %.3f'%(i,threshVal,inequal,weightedError))
				if weightedError<minError:
					minError=weightedError #保存当前最小的错误率
					bestClasEst=predictedVals.copy() #预测类别
					#保存该层决策树
					bestStump['dim']=i
					bestStump['thresh']=threshVal
					bestStump['ineq']=inequal
	return bestStump,minError,bestClasEst

#基于单层决策树的AdaBoost训练过程
def adaBoostTrainDS(dataArr,classLabels,numIt=40):
	weakClassArr=[]
	m=shape(dataArr)[0]
	D=mat(ones((m,1))/m)
	aggClassEst=mat(zeros((m,1)))
	#迭代
	for i in range(numIt):
		bestStump,error,classEst=buildStump(dataArr,classLabels,D) #找到最佳单层决策树
		print('D:',D.T)
		#根据误差率计算alpha
		alpha=float(0.5*log((1.0-error)/max(error,1e-16))) #1e-16防止零溢出
		bestStump['alpha']=alpha #将alpha加入字典
		print('bestStump',bestStump)
		weakClassArr.append(bestStump)
		print('weakClassArr',weakClassArr)
		print('classEst:',classEst.T)
		#计算新的权重D(公式推导，统计学)
		expon=multiply(-1*alpha*mat(classLabels).T,classEst)
		D=multiply(D,exp(expon))
		D=D/D.sum()
		#更新类别估计值
		aggClassEst+=alpha*classEst
		print('aggClassEst:',aggClassEst.T)
		aggErrors=multiply(sign(aggClassEst)!=mat(classLabels).T,ones((m,1)))
		errorRate=aggErrors.sum()/m
		print('total error:',errorRate)
		if errorRate==0.0:
			break
	return weakClassArr		

#分类
# 遍历classifierArr中的所有弱分类器，并基于stumpClassify()对每个分类器得到一个类别的估计值。
# 输出的类别估计值乘上该单层决策树的alpha权重然后累加到aggClassEst上，就完成了这一过程。最后
# 程序返回aggClassEst的符号，即如果aggClassEst大于0则返回1，而如果小于0则返回-1
def adaClassify(datToClass,classifyArr):
	dataMatrix=mat(datToClass)
	m=shape(dataMatrix)[0]
	aggClassEst=mat(zeros((m,1)))
	for i in range(len(classifierArr)):
		#classEst指预测出的当前特征的分类
		classEst=stumpClassify(dataMatrix,classifyArr[i]['dim'],classifyArr[i]['thresh'],classifyArr[i]['ineq'])
		aggClassEst+=classifierArr[i]['alpha']*classEst
	return sign(aggClassEst)	

dataMat,classLabels=loadSimpData()	
# D=mat(ones((5,1))/5) #初始化训练数据的权值分布。如果有N个样本，则每一个训练样本最开始时都被赋予相同的权值：1/N
# #输入数据集，分类，权重
# buildStump(dataMat,classLabels,D)	
classifierArr=adaBoostTrainDS(dataMat,classLabels,30)
#print('classifierArr',classifierArr)
adaClassify([0,0],classifierArr)	

#在一个难数据集上应用adaboost
def loadDataSet(filename):
	numFeat=len(open(filename).readline().split('\t'))
	dataMat=[];labelMat=[]
	fr=open(filename)
	for line in fr.readlines():
		lineArr=[]
		curLine=line.strip().split('\t')
		for i in range(numFeat):
			lineArr.append(curLine[i])
		dataMat.append(lineArr)
		labelMat.append(float(curLine[-1]))
	return dataMat,labelMat				
