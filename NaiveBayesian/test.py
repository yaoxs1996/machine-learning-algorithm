import bayes

listOPosts, listClasses = bayes.loadDataSet()
myVocabList = bayes.createVocabList(listOPosts)
print(myVocabList)
print(bayes.setOfWord2Vec(myVocabList, listOPosts[0]))
print(bayes.setOfWord2Vec(myVocabList, listOPosts[3]))