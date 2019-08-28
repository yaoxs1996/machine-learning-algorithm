import kNN
import matplotlib.pyplot as plt

group, labels = kNN.createDataSet()
#print(group)
#print(labels)

# 预测数据所在分类
kNN.classify0([0, 0], group, labels, 3)

datingDataMat, datingLabels = kNN.file2matrix('datingTestSet2.txt')
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datingDataMat[:,1], datingDataMat[:,2])
plt.show()