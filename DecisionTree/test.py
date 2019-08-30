import trees

myDat, labels = trees.createDataSet()
'''
# print(myDat)
print(trees.calcShannonEnt(myDat))
# 熵越高，则混合的数据也就越多
# myDat[0][-1] = 'maybe'
print(trees.calcShannonEnt(myDat))

print(trees.splitDataSet(myDat, 0, 1))
print(trees.splitDataSet(myDat, 0, 0))
'''

# 划分数据集的测试代码
'''
print(trees.chooseBestFeatureToSplit(myDat))        #表明第0个特征是最好用于划分数据集的特征
print(myDat)
'''

# 创建树的测试代码
myTree = trees.createTree(myDat, labels)
print(myTree)