from numpy import *
import math

# 词表到向量的转换函数
def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
            ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
            ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
            ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
            ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
            ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]       # 1代表侮辱性文字，0代表正常言论
    return postingList, classVec

def createVocabList(dataSet):
    vocabSet = set([])      # 创建一个空集
    for document in dataSet:
        vocabSet = vocabSet | set(document)     # 创建两个集合的并集
    return list(vocabSet)

def setOfWord2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)        # 创建一个其中所含元素都为0的向量
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" %word)
    return returnVec
    
# 朴素贝叶斯分类器训练函数
# 文档矩阵trainMatrix，文档类别标签向量trainCategory
def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    # 初始化概率
    '''
    p0Num = zeros(numWords)
    p1Num = zeros(numWords)
    p0Denom = 0.0
    p1Denom = 0.0
    '''
    # 降低概率为0的影响，将词的出现次数初始化为1，分母初始化为2
    p0Num = ones(numWords)
    p1Num = ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            # 向量相加
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    # 对每个元素做除法
    '''
    p1Vect = p1Num / p1Denom
    p0Vect = p0Num / p0Denom
    '''
    # 取log，防止下溢。log每次只处理一个数字，对矩阵操作，需要进行遍历
    # p1Vect = math.log(p1Num / p1Denom)
    # p0Vect = math.log(p0Num / p0Denom)
    a = p1Num / p1Denom
    b = p0Num / p0Denom
    p1Vect = [math.log(x) for x in a]
    p0Vect = [math.log(x) for x in b]
    return p0Vect, p1Vect, pAbusive

# 朴素贝叶斯分类函数
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + math.log(pClass1)      # 元素相乘
    p0 = sum(vec2Classify * p0Vec) + math.log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWord2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(array(trainMat), array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWord2Vec(myVocabList, testEntry))
    print(testEntry, ' classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWord2Vec(myVocabList, testEntry))
    print(testEntry, ' classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))