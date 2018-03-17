import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.ensemble import ExtraTreesClassifier

data = fetch_olivetti_faces()

def importance(n_estimators=500, max_features=128, n_jobs=3, random_state=0):
	x = data.images.reshape((len(data.images), -1))
	y = data.target
	forest = ExtraTreesClassifier(n_estimators=n_estimators, max_features=max_features, n_jobs=n_jobs, random_state=random_state)
	forest.fit(x, y)
	importance = forest.feature_importances_
	importance = importance.reshape(data.images[0].shape)
	plt.matshow(importance, cmap=plt.cm.hot)
	plt.show()

importance()