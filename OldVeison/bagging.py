from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets

bcls = BaggingClassifier(DecisionTreeClassifier(), max_samples=0.5, max_features=0.5, n_estimators=50)
x,y = datasets.make_blobs(n_samples=8000, centers=2, random_state=0, cluster_std=4)
bcls.fit(x,y)
print(bcls.score(x,y))