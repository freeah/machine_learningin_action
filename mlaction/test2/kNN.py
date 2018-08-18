from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt
from os import listdir

#k邻近算法
# 对未知类别属性的数据集中的每个点依次执行以下操作：
# 1、计算已知类别数据集中的点与当前点的距离
# 2、按照距离递增次序排序
# 3、选取与当前点距离最小的k个点
# 4、确定当前k个点所在类别的出现频率
# 5、返回前k个点出现频率最高的类别作为当前点的预测分类
# inX表示当前数据array dataSet表示所有数据集  labels表示数据集每条数据的类型集   k表示最小的k个点

def createDataSet():
    group=array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels=['A','A','B','B']
    return group,labels

def classify0(inX,dataSet,labels,k):
    #print(dataSet.shape)
    dataSetSize=dataSet.shape[0]  #获取dataSet的行数
    #print(dataSetSize)
    #print(tile(inX,(4,1)))  #numpy 将原矩阵纵向复制  https://www.jianshu.com/p/9519f1984c70
    diffMat=tile(inX,(dataSetSize,1))-dataSet
    #print(diffMat)
    sqDiffMat=diffMat**2
    #print(sqDiffMat)
    sqDistances=sqDiffMat.sum(axis=1)
    #print(sqDistances)
    distances=sqDistances**0.5
    #print(distances)
    #以上使用欧氏距离计算向量点之间的距离，各点差的平方和开根号
    sortedDistIndicies=distances.argsort()  #将distances中的元素从小到大排列，提取其对应的index(索引)，然后输出到sortedDistIndicies
    #print(sortedDistIndicies)
    classCount={}
    #选择距离最小的k个点
    for i in range(k):
        voteIlabel=labels[sortedDistIndicies[i]]  #B B  A
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1   #.get()获取指定键的值，如果值不在字典内，则返回设置的默认值
    #print(classCount) #{B:2,A:1} 
    #items() 函数以列表返回可遍历的(键, 值) 元组数组  
    #print(classCount.items())
    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)  #倒序排序
    #print(sortedClassCount)
    return sortedClassCount[0][0]

group,labels=createDataSet()
#print(group,labels)
#print(classify0([0,0],group,labels,3))  

# 使用k邻近算法改进约会网站的配对效果
# 将文本记录到转换numpy的解析程序
def file2matrix(filename):
    fr=open(filename)  #读取文件
    #readlines() 方法用于读取所有行(直到结束符 EOF)并返回列表
    arrayOLines=fr.readlines()  
    #print(arrayOLines) 
    numberOfLines=len(arrayOLines)  #获取文件的行数
    #numpy.zeros(shape, dtype=float, order='C')
    #返回给定形状和类型的新数组，用0填充
    returnMat=zeros((numberOfLines,3))
    #print(returnMat)
    classLabelVector=[]
    index=0
    for line in arrayOLines:
        line=line.strip() #Python strip() 方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列
        listFormLine=line.split('\t')
        returnMat[index,:]=listFormLine[0:3]
        classLabelVector.append(int(listFormLine[-1]))
        index+=1
    return returnMat,classLabelVector    

#分析数据，使用Matplotlib创建散点图
# fig=plt.figure()
# ax=fig.add_subplot(111)
# ax.scatter(datingDataMat[:,1],datingDataMat[:,2],15.0*array(datingLabels),15.0*array(datingLabels))
# plt.show()

#准备数据，归一化数值(第一列数值远远大于第二列第三列数值，为了减小这种影响)
#newVal=(oldVal-min)/(max-min)
def autoNorm(dataSet):
    minVals=dataSet.min(0) #找到每一列的最小值numpy
    #print(minVals)
    maxVals=dataSet.max(0)
    #print(maxVals)
    ranges=maxVals-minVals
    normDataSet=zeros(shape(dataSet))
    #print(normDataSet)
    m=dataSet.shape[0]
    normDataSet=dataSet-tile(minVals,(m,1)) #相应每列的每个数减去当前列的最小值
    normDataSet=normDataSet/tile(ranges,(m,1)) #差除以ranges
    return normDataSet,ranges,minVals


#测试算法，作为完整程序验证分类器
def datingClassTest():
    hoRatio=0.10
    datingDataMat,datingLabels=file2matrix('datingTestSet2.txt')  #获取前三列，最后一列的数据
    normMat,ranges,minVals=autoNorm(datingDataMat)  #数据归一化
    m=normMat.shape[0]  #1000
    numTestVecs=int(m*hoRatio)  #100
    errorCount=0.0
    for i in range(numTestVecs):
        classifierResult=classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print('the classifier came back with:%d,the real answer is:%d' %(classifierResult,datingLabels[i]))
        if classifierResult!=datingLabels[i]:
            errorCount+=1.0
    print('the total error rate is:%f' %(errorCount/float(numTestVecs)))        

datingClassTest() 

#约会网站预测函数
def classifyPerson():
    resultList=['not at all','in small doses','in large doses']
    percentTats=float(input('percentage of time spent playing video games?'))
    iceCream=float(input('liters of ice cream consumed per year?'))
    ffMiles=float(input('freguent flier miles earned per year?'))
    datingDataMat,datingLabels=file2matrix('datingTestSet2.txt')
    normMat,ranges,minVals=autoNorm(datingDataMat)
    inArr=array([ffMiles,percentTats,iceCream])
    classifierResult=classify0((inArr-minVals)/ranges,normMat,datingLabels,3)
    print('you will properly like this person:',resultList[classifierResult-1])

#classifyPerson()    

#手写识别系统
#准备数据，将图像转换为测试向量
def img2vector(filename):
    returnVect=zeros((1,1024))
    fr=open(filename)
    for i in range(32):
        lineStr=fr.readline()
        for j in range(32):
            returnVect[0,32*i+j]=int(lineStr[j])
    return returnVect        

#print(img2vector('digits/trainingDigits/0_13.txt'))
#手写数字识别系统测试代码
def handwritingClassTest():
    hwLabels=[]
    #读取当前文件夹下的所有文件
    trainingFileList=listdir('digits/trainingDigits')
    #print(trainingFileList)
    m=len(trainingFileList)
    trainingMat=zeros((m,1024))
    for i in range(m):
        fileNameStr=trainingFileList[i]
        fileStr=fileNameStr.split('.')[0]
        classNumStr=int(fileStr.split('_')[0]) #获取当前数字
        hwLabels.append(classNumStr)
        trainingMat[i,:]=img2vector('digits/trainingDigits/%s'%fileNameStr)
    testFileList=listdir('digits/testDigits')
    errorCount1=0.0
    mTest=len(testFileList)
    for i in range(mTest):
        fileNameStr=testFileList[i]
        fileStr=fileNameStr.split('.')[0]
        classNumStr=int(fileStr.split('_')[0])
        vectorUnderTest=img2vector('digits/testDigits/%s'%fileNameStr)
        classifierResult=classify0(vectorUnderTest,trainingMat,hwLabels,3) #代入函数，返回测试结果
        print('the classifier came back with:%d,the real answer is:%d'%(classifierResult,classNumStr))
        if classifierResult!=classNumStr:
            errorCount1+=1.0
    print('the total number of errors is %d' %errorCount1)
    print('the totla error rate is:%f'%(errorCount1/float(mTest)))        

handwritingClassTest()