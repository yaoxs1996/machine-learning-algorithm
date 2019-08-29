import trees

myDat, labels = trees.createDataSet()
# print(myDat)
print(trees.calcShannonEnt(myDat))
# 熵越高，则混合的数据也就越多
myDat[0][-1] = 'maybe'
print(trees.calcShannonEnt(myDat))

print(trees.splitDataSet(myDat, 0, 1))
print(trees.splitDataSet(myDat, 0, 0))