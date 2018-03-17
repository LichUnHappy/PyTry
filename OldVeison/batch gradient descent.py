import numpy as np
import random
import matplotlib.pyplot as plt

def gradientDecent(x, y, alpha, numIteration):
	xTrans = x.transpose()
	m, n = np.shape(x)
	theta = np.ones(n)
	for i in range(0, numIteration):
		hwx = np.dot(x, theta)
		loss = hwx - y
		cost = np.sum(loss ** 2) / (2 * m)
		print("Iteration %d | Cost: %f " % (i, cost))
		gradient = np.dot(xTrans, loss) / m
		theta = theta - alpha * gradient
	return theta

def genData(numPoints, bias, variance):
	x = np.zeros(shape=(numPoints, 2))
	y = np.zeros(shape=numPoints)
	for i in range(0, numPoints):
		x[i][0] = 1
		x[i][1] = i
		y[i] = (i + bias) + random.uniform(0, 1) * variance
	return x, y

def plotData(x, y, theta):
	plt.scatter(x[:,1], y)
	plt.plot(x[:,1], [theta[0] + theta[1] * xi for xi in x[:,1]])
	plt.show()

x, y = genData(20, 1, 20)
iterations = 1000
alpha = 0.001
theta = gradientDecent(x, y, alpha, iterations)
plotData(x, y, theta)