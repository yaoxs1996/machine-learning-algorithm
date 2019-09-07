import bayes

listOPosts, listClasses = bayes.loadDataSet()
myVocabList = bayes.createVocabList(listOPosts)
'''
print(myVocabList)
print(bayes.setOfWord2Vec(myVocabList, listOPosts[0]))
print(bayes.setOfWord2Vec(myVocabList, listOPosts[3]))
'''

# 朴素贝叶斯分类器
'''
trainMat = []
for postinDoc in listOPosts:
    trainMat.append(bayes.setOfWord2Vec(myVocabList, postinDoc))
p0V, p1V, pAb = bayes.trainNB0(trainMat, listClasses)
print(pAb)
print(p0V)
print(p1V)
'''

bayes.testingNB()