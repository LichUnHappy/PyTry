# 线性规划
# from scipy.optimize import linprog
# objective = [-1, -1]
# con1 = [[2,1], [1,2]]
# con2 = [4, 3]
# res = linprog(objective, con1, con2)
# print(res)

# 最小化
# import numpy as np 
# from scipy.optimize import minimize
# def rosen(x):
# 	return sum(100.0 * (x[1:] - x[:1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)

# def nMin(funct, x0):
# 	return(minimize(rosen, x0, method='nelder-mead', options={'xtol':1e-8, 'disp':True}))

# x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])

# nMin(rosen, x0)

#Scikit-learn
# from sklearn import datasets
# iris = datasets.load_iris()
# iris_X = iris.data
# iris_Y = iris.target
# print(iris_X.shape)
# print(iris.DESCR)


# k-nn KNeighbors
# from sklearn.neighbors import KNeighborsClassifier as knn
# from sklearn import datasets
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.colors import ListedColormap

# def knnDemo(X, y, n):

# 	# create the classifier and fit it to the data
# 	res = 0.05
# 	k1 = knn(n_neighbors=n, p=2, metric='minkowski')
# 	k1.fit(X,y)

# 	# set up the grid
# 	x1_min, x1_max = X[:,0].min() - 1, X[:,0].max() + 1
# 	x2_min, x2_max = X[:,1].min() - 1, X[:,0].max() + 1
# 	xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, res), np.arange(x2_min, x2_max, res))

# 	# make the prediction
# 	z = k1.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
# 	z = z.reshape(xx1.shape)

# 	# create the color map
# 	cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
# 	cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# 	# plot the decision surface
# 	plt.contourf(xx1, xx2, z, alpha=0.4, cmap=cmap_light)
# 	plt.xlim(xx1.min(), xx1.max())
# 	plt.ylim(xx2.min(), xx2.max())

# 	#plot the sample
# 	for idx, c1 in enumerate(np.unique(y)):
# 		plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)

# 	plt.show()

# iris = datasets.load_iris()
# X1 = iris.data[:, 0:3:2]
# X2 = iris.data[:, 0:2]
# X3 = iris.data[:, 1:3]
# y = iris.target
# knnDemo(X2, y, 15)

# 线性回归 
# from sklearn import linear_model
# clf = linear_model.LinearRegression()
# clf.fit([[0,0], [1,1], [2,2]], [0, 1, 2])
# print(clf.coef_)


# 岭回归
# from sklearn.linear_model import Ridge
# import numpy as np

# def ridgeReg(alpha):

# 	n_samples, n_features = 10, 5
# 	y = np.random.randn(n_samples)
# 	X = np.random.randn(n_samples, n_features)
# 	clf = Ridge(.001)
# 	res = clf.fit(X, y)
# 	return(res)

# res = ridgeReg(0.001)
# print(res.coef_)
# print(res.intercept_)

# PCA
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.decomposition import KernelPCA
# from sklearn.datasets import make_circles
# np.random.seed()
# X, y = make_circles(n_samples=400, factor=.3, noise=.05)
# kpca = KernelPCA(kernel='rbf', gamma=10)
# X_kpca = kpca.fit_transform(X)


# plt.figure()

# plt.subplot(2, 2, 1, aspect='equal')
# plt.title("Original space")
# reds = y == 0
# blues = y == 1
# plt.plot(X[reds, 0], X[reds, 1], "ro")
# plt.plot(X[blues, 0], X[blues, 1], "bo")
# plt.xlabel("$x_1$")
# plt.ylabel("$x_2$")

# plt.subplot(2, 2, 3, aspect='equal')
# plt.plot(X_kpca[reds, 0], X_kpca[reds, 1], "ro")
# plt.plot(X_kpca[blues, 0], X_kpca[blues, 1], "bo")
# plt.title("Projection by KPCA")
# plt.xlabel("1st principal component in space induced by $\phi$")
# plt.ylabel("2nd component")
# plt.subplots_adjust(0.02, 0.10, 0.98, 0.94, 0.04, 0.35)
# plt.show()

# 交叉验证
# from sklearn.cross_validation import train_test_split
# from sklearn import datasets
# from sklearn import svm
# from sklearn import cross_validation
# iris = datasets.load_iris()
# x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.4, random_state=0)
# clf = svm.SVC(kernel='linear', C=1).fit(x_train, y_train)
# scores = cross_validation.cross_val_score(clf, x_train, y_train, cv=5)
# print("Accurency: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))