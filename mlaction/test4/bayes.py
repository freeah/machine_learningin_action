from numpy import *
#词表到向量转换函数
def loadDataSet():
	postingList=[
	['my','dog','has','flea','problems','help','please'],
	['maybe','not','take','him','to','dog','park','stupid'],
	['my','dalmation','is','so','cute','I','love','him'],
	['stop','posting','stupid','worthless','garbage'],
	['mr','licks','ate','my','steak','how','to','stop','him'],
	['quit','buying','worthless','dog','food','stupoid']]
	classVec=[0,1,0,1,0,1] #1代表侮辱性的文字  0代表正常言论
	return postingList,classVec

# a=[1,2,3,4,5]
# b=[6,7,8,9,10]
# print(set(a)|set(b))  #{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

#创建一个包含在所有文档中出现的不重复词的列表
def createVocabList(dataSet):
	vocabSet=set([])
	for document in dataSet:
		vocabSet=vocabSet | set(document) #并集
	return list(vocabSet)

#print([0]*len([1,2,3])) #[0,0,0]

 #输入列表单词如果在词汇表里面，则新数组当前位置为1，否则为0	
def setOfWords2Vec(vocabList,inputSet):
	returnVec=[0]*len(vocabList)  #vocabList等长的列表用0填充
	for word in inputSet:
		if word in vocabList:
			returnVec[vocabList.index(word)]=1 #.index()返回包含字符串的索引值
		else:
			print('the word:%s is not in my Vocabulary!'%word)
	return returnVec			

#朴素贝叶斯分类器训练函数
def trainNB0(trainMatrix,trainCategory):
	numTrainDocs=len(trainMatrix) #文档数
	print(numTrainDocs)
	numWords=len(trainMatrix[0]) #所有词汇数
	print(numWords)
	pAbusive=sum(trainCategory)/float(numTrainDocs)  #侮辱性文档数/总的文档数
	p0Num=ones(numWords)
	p1Num=ones(numWords)
	p0Denom=2.0
	p1Denom=2.0
	#计算每个分类中，相应单词出现的概率
	#统计每个单词的出现次数，出现次数/词汇总数=每个单词出现的概率
	#循环遍历trainMatrix中的所有文档，一旦某个词语在某个文档中出现，则该词对应的个数就+1，
	#而且在所有文档中，该文档总词数也相应+1
	for i in range(numTrainDocs):
		print('文档当前类别：%s'%trainCategory[i])
		print(trainMatrix[i])
		if trainCategory[i]==1:
			p1Num+=trainMatrix[i]
			p1Denom+=sum(trainMatrix[i])
			# print(p1Num)
			# print('所有词条数',p1Denom)
		else:
			print(p0Num)
			p0Num+=trainMatrix[i]
			p0Denom+=sum(trainMatrix[i])
			print(p0Num)
			print('所有词条数',p0Denom)		
	p1Vect=p1Num/p1Denom
	p0Vect=p0Num/p0Denom
	return p0Vect,p1Vect,pAbusive			

#朴素贝叶斯分类函数
def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
	print('vec2Classify',vec2Classify)
	print('p0Vec',p0Vec)
	print('p1Vec',p1Vec)
	print('pClass1',pClass1)
	print(vec2Classify*p1Vec)
	#这里没有直接计算P（x，y|C1）P（C1），而是取其对数
    #这样做也是防止概率之积太小，导致四舍五入为0
	p1=sum(vec2Classify*p1Vec)+log(pClass1) 
	p0=sum(vec2Classify*p0Vec)+log(1.0-pClass1)
	if p1>p0:
		return 1
	else:
		return 0

def testingNB():
	listOPosts,listClasses=loadDataSet()  #返回所有文档词汇，文档分类
	myVocabList=createVocabList(listOPosts) #返回所有文档中不重复的词汇表
	print('myVocabList',myVocabList)
	trainMat=[]	
	for postinDoc in listOPosts:
		print(postinDoc)
		trainMat.append(setOfWords2Vec(myVocabList,postinDoc))	
	print('trainMat',trainMat)	
	print('listClasses',listClasses)
	p0V,p1V,pAb=trainNB0(array(trainMat),array(listClasses)) #计算单词在两个分类中出现的概率和p(c1)
	
	testEntry=['love','my','dalmation']
	thisDoc=array(setOfWords2Vec(myVocabList,testEntry))
	print(testEntry,'classified as:',classifyNB(thisDoc,p0V,p1V,pAb))
	testEntry=['stupid','garbage']
	thisDoc=array(setOfWords2Vec(myVocabList,testEntry))
	print(testEntry,'classified as:',classifyNB(thisDoc,p0V,p1V,pAb))	


#testingNB()

#词袋模型
def bagOfWords2VecMN(vocabList,inputSet):
	returnVec=[0]*len(vocabList)
	for word in inputSet:
		if word in vocabList:
			returnVec[vocabList.index(word)]+=1
	return returnVec	
	
#使用朴素贝叶斯过滤垃圾邮件
#收集数据：提供文本文件
#准备数据：将文本文件解析成词条向量
#分析数据：检查词条，确保解析的正确性
#训练算法：使用之前建立的trainNB0()函数
#使用算法：构建一个完整的程序对一组文档进行分类，将错分的文档输出到屏幕上

#切分文本
def textParse(bigString):
	import re
	listOfTokens=re.split(r'\W*',bigString)
	return [tok.lower() for tok in listOfTokens if len(tok)>2]

#完整的垃圾邮件测试函数
def spamTest():
	docList=[];classList=[];fullText=[]
	#读取文件，生成词汇表
	for i in range(1,26):
		wordList=textParse(open('email/spam/%d.txt'%i).read())
		docList.append(wordList)
		fullText.extend(wordList)
		classList.append(1)
		#在读取Walden.txt文本时，出现了“UnicodeDecodeError: 'gbk' codec can't decode byte 0xbf in position 2: illegal multibyte sequence”错误提示。
		wordList=textParse(open('email/ham/%d.txt'%i,encoding='gb18030',errors='ignore').read())
		docList.append(wordList)
		fullText.extend(wordList)
		classList.append(0)
	print('docList',docList)
	print('fullText',fullText)
	print('classList',classList)  #类别
	vocabList=createVocabList(docList) #生成不包含重复单词的词汇表
	trainingSet=list(range(50));testSet=[]
	#取出十篇文档用来测试，
	#其余40篇文档用来训练，求每个单词在每类中的概率
	for i in range(10):
		#生成0-len(trainingSet)之间的随机数，包含第一个数，不包含最后一个数
		randIndex=int(random.uniform(0,len(trainingSet)))
		testSet.append(randIndex) #随机产生的十篇文档
		del(trainingSet[randIndex]) #在测试文档中去掉那十篇文档
	print('testSet',testSet)	
	print('trainingSet',trainingSet)	
	trainMat=[];traingClasses=[]
	for docIndex in trainingSet:
		trainMat.append(setOfWords2Vec(vocabList,docList[docIndex]))
		traingClasses.append(classList[docIndex])
	p0V,p1V,pSpam=trainNB0(array(trainMat),array(traingClasses)) #训练求概率
	errorCount=0
	#测试计算错误率
	for docIndex in testSet:
		wordVector=setOfWords2Vec(vocabList,docList[docIndex])
		if classifyNB(array(wordVector),p0V,p1V,pSpam)!=classList[docIndex]:
			errorCount+=1
	print('the error rate is:',float(errorCount)/len(testSet))					

spamTest()	