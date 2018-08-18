from math import log
#计算给定数据集的熵
def calcShannonEnt(dataSet):
	numEntries=len(dataSet)
	#为所有可能分类创建字典，统计各个类型的数目
	labelCounts={}
	for featVec in dataSet:
		currentLabel=featVec[-1]
		if currentLabel not in labelCounts.keys():
			labelCounts[currentLabel]=0
		labelCounts[currentLabel]+=1
	#代入公式计算熵
	shannonEnt=0.0
	for key in labelCounts:
		prob=float(labelCounts[key])/numEntries
		shannonEnt-=prob*log(prob,2)
	return shannonEnt
	
#自定义数据计算熵
dataSet=[[1,1,'yes'],[1,1,'yes'],[1,0,'no'],[0,1,'no'],[0,1,'no']]
#print(calcShannonEnt(dataSet))		

#按照给定特征划分数据集
#三个输入参数，待划分的数据集，划分数据集的特征，特征的返回值
def splitDataSet(dataSet,axis,value):
	retDataSet=[]
	#如果特征的值等于指定的值，则返回除特征值以外的值
	for featVec in dataSet:
		if featVec[axis]==value:
			reduceFeatVec=featVec[:axis]
			reduceFeatVec.extend(featVec[axis+1:])
			retDataSet.append(reduceFeatVec)
	return retDataSet			

#print(splitDataSet(dataSet,0,1)) #[[1, 'yes'], [1, 'yes'], [0, 'no']]
#print(splitDataSet(dataSet,1,0)) #[[1, 'no']]

#选择最好的数据集划分方式
def chooseBestFeatureToSplit(dataSet):
	numFeatures=len(dataSet[0])-1 #当前数据集包含多少特征属性
	baseEntropy=calcShannonEnt(dataSet) #初始数据集的熵
	bestInfoGain=0.0
	bestFeature=-1
	#循环数据集的特征
	for i in range(numFeatures):
		featList=[example[i] for example in dataSet] #某一特征值的集合
		uniqueVals=set(featList) #创建一个无序不重复的元素集
		newEntropy=0.0
		#计算每种划分方式的信息熵
		for value in uniqueVals:
			subDataSet=splitDataSet(dataSet,i,value)
			prob=len(subDataSet)/float(len(dataSet))
			newEntropy+=prob*calcShannonEnt(subDataSet)
		#计算最好的信息增益，信息增益是熵的减少或者是数据无序度的减少
		infoGain=baseEntropy-newEntropy	
		if infoGain>bestInfoGain:
			bestInfoGain=infoGain
			bestFeature=i
	return bestFeature
			
print(chooseBestFeatureToSplit(dataSet))		