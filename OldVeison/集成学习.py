from sklearn import cross_validation
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import VotingClassifier
from sklearn import datasets

def vclas(w1, w2, w3, w4):

	x, y = datasets.make_classification(n_features=10, n_informative=4, n_samples=500, n_clusters_per_class=5)
	xtrain, xtest, ytrain, ytest = cross_validation.train_test_split(x, y, test_size=0.4)

	clf1 = LogisticRegression(random_state=123)
	clf2 = GaussianNB()
	clf3 = RandomForestClassifier(n_estimators=10, bootstrap=True, random_state=123)
	clf4 = ExtraTreesClassifier(n_estimators=10, bootstrap=True, random_state=123)

	clfes = [clf1, clf2, clf3, clf4]

	eclf = VotingClassifier(estimators=[('1r', clf1), ('gnb', clf2), ('rf', clf3), ('et', clf4)], 
							voting='soft', weights=[w1, w2, w3, w4])

	[c.fit(xtrain, ytrain) for c in (clf1, clf2, clf3, clf4, eclf)]

	N = 5
	ind = np.arange(N)
	width = 0.3
	ax = plt.subplot()


	for i, clf in enumerate(clfes):
		print(clf, i)
		p1 = ax.bar(i, clfes[i].score(xtrain, ytrain), width=width, color='black')
		p2 = ax.bar(i + width, clfes[i].score(xtest, ytest), width=width, color='grey')

	ax.bar(len(clfes) + width, eclf.score(xtrain, ytrain), width=width, color='black')
	ax.bar(len(clfes) + width * 2, eclf.score(xtest, ytest), width=width, color='grey')
	plt.axvline(3.8, color='k', linestyle='dashed')
	ax.set_xticks(ind + width)
	ax.set_xticklabels(['LogisticRegression', 'GaussianNB',
					    'RandomForestClassifier', 'ExtraTreesClassifier',
					    'VotingClassifier'], rotation=40, ha='right')
	plt.title('Training and test score for different classfiers')
	plt.legend([p1[0], p2[0]], ['Training', 'Test'], loc='lower right')
	plt.show()

vclas(1, 2, 3, 4)