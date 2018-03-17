
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

# 正态分布
# def normal(mean=0, var=1):
# 	sigma = np.sqrt(var)
# 	x = np.linspace(-3,3,100)
# 	plt.plot(x, mlab.normpdf(x, mean, sigma))
# 	plt.show()

# normal(0, 1)

# 二项分布
# from scipy.stats import binom
# def binomial(x=10, n=10, p=0.5):
# 	x = range(x)
# 	rv = binom(n, p)
# 	plt.vlines(x, 0, (rv.pmf(x)), colors='k', linestyles='-')
# 	plt.show()

# binomial()

# 泊松分布
# from scipy.stats import poisson
# def pois(x=1000):
# 	xr = np.arange(x)
# 	ps = poisson(xr)
# 	plt.plot(ps.pmf(x/2))
# 	plt.show()
# pois()

# 累积分布函数
# import scipy.stats as stats
# def cdf(s1=50, s2=0.2):
# 	x = np.linspace(0, s2 * 100, s1 * 2)
# 	cd = stats.binom.cdf
# 	plt.plot(x, cd(x, s1, s2))
# 	plt.show()

# cdf()