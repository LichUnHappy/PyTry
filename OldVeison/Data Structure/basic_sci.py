# from scipy.optimize import linprog

# objective = [-1, -1]
# con1 = [[2,1], [1,2]]
# con2 = [4, 3]
# res = linprog(objective, con1, con2)
# print(res)

# import numpy as np 
# from scipy.optimize import minimize
# def rosen(x):
# 	return sum(100.0 * (x[1:] - x[:1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)

# def nMin(funct, x0):
# 	return(minimize(rosen, x0, method='nelder-mead', options={'xtol':1e-8, 'disp':True}))

# x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])

# nMin(rosen, x0)

#Scikit-learn
from sklearn import datasets
iris = datasets.load_iris()
iris_X = iris.data
iris_Y = iris.target
# print(iris_X.shape)
# print(iris.DESCR)

from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def knnDemo(X, y, n):

	# create the classifier and fit it to the data
	res = 0.05
	k1 = knn(n_neighbors=n, p=2, metric='minkowski')
	k1.fit(X,y)

	# set up the grid
	x1_min, x1_max = X[:,0].min() - 1, X[:,0].max() + 1
	x2_min, x2_max = X[:,1].min() - 1, X[:,0].max() + 1
	xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, res), np.arange(x2_min, x2_max, res))

	# make the prediction
	z = k1.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
	z = z.reshape(xx1.shape)

	# create the color map
	cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
	cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

	# plot the decision surface
	plt.contourf(xx1, xx2, z, alpha=0.4, cmap=cmap_light)
	plt.xlim(xx1.min(), xx1.max())
	plt.ylim(xx2.min(), xx2.max())

	#plot the sample
	for idx, c1 in enumerate(np.unique(y)):
		plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)

	plt.show()

iris = datasets.load_iris()
X1 = iris.data[:, 0:3:2]
X2 = iris.data[:, 0:2]
X3 = iris.data[:, 1:3]
y = iris.target
knnDemo(X2, y, 15)