# import numpy as np 
# import matplotlib.pyplot as plt 

# 折线图
# x = np.arange(0., 5., 0.2)
# plt.plot(x, x**4, 'r', x, x*90, 'bs', x, x**3, 'g^')
# plt.show()


#组图
# x1 = np.arange(0., 5., 0.2)
# x2 = np.arange(0., 5., 0.2)


# plt.figure(1)

# plt.subplot(211)
# plt.plot(x1, x1**4, 'r', x1, x1*90, 'bs', x1, x1**3, 'g^', linewidth=2.0)

# plt.subplot(212)
# plt.plot(x2, np.cos(2*np.pi*x2), 'k')
# plt.show()


# 直方图
# mu, sigma = 100, 15
# x = mu + sigma * np.random.randn(1000)
# n, bins, patches = plt.hist(x, 10, normed=1, facecolor='g')
# plt.xlabel('Frequency')
# plt.ylabel('Probability')
# plt.title('Histogram Example')
# plt.text(40, .028, 'mean=100, std.dev.=15')
# plt.axis([40, 160, 0, 0.03])
# plt.grid(True)
# plt.show()

# 散点图
# N = 1000
# x = np.random.rand(N)
# y = np.random.rand(N)
# colors = np.random.rand(N)
# # colors = ('r', 'b', 'g')
# area = np.pi * (10 * np.random.rand(N))**2
# plt.scatter(x, y, s=area, c=colors, alpha=0.5)
# plt.show()

# 三维图
# import matplotlib as mpl 
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import cm

# mpl.rcParams['legend.fontsize'] = 10

# fig = plt.figure()
# ax = fig.gca(projection='3d')

# theta = np.linspace(-3 * np.pi, 6 * np.pi, 100)
# z = np.linspace(-2, 2, 100)
# r = z ** 2 + 1
# x = r * np.sin(theta)
# y = r * np.cos(theta)
# # ax.plot(x, y, z)

# theta2 = np.linspace(-3 * np.pi, 6 * np.pi, 20)
# z2 = np.linspace(-2, 2, 20)
# r2 = z2 ** 2
# x2 = r2 * np.sin(theta2)
# y2 = r2 * np.cos(theta2)
# # ax.scatter(x2, y2, z2, c='r')


# x3 = np.arange(-5, 5, 0.25)
# y3 = np.arange(-5, 5, 0.25)
# x3, y3 = np.meshgrid(x3, y3)
# R = np.sqrt(x3  ** 2 + y3 ** 2)
# z3 = np.sin(R)
# surf = ax.plot_surface(x3, y3, z3, rstride=1, cstride=1, cmap=cm.Greys_r, linewidth=0, antialiased=False)
# ax.set_zlim(-2, 2)
# plt.show()

