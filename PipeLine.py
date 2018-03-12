from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.externals import joblib

iris = load_iris()
x, y = iris.data, iris.target

pca = PCA(n_components=2)
selection = SelectKBest(k=1)
combined_features = FeatureUnion([("pca", pca),
                                ("univ_select", selection)])
# print(combined_features)

svm = SVC(kernel="linear")

pipeline = Pipeline([("features", combined_features), ("svm", svm)])
pipeline.fit(x,y)

pipeline.predict(x)
joblib.dump(pipeline, 'iris-pipeline.pkl')

print("Clear!")