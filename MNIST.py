from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

digits = datasets.load_digits()

X_digits = digits.data
Y_digits = digits.target

# 设置样本分割点
alpha = 0.9
n_samples_train = round(alpha*len(X_digits))
n_samples_test = round((1-alpha)*len(X_digits))


X_train = X_digits[:n_samples_train]
Y_train = Y_digits[:n_samples_train]
X_test = X_digits[:-n_samples_test]
Y_test = Y_digits[:-n_samples_test]

# 逻辑斯底回归
# model = LogisticRegression()

# ｋ近邻分类
model = KNeighborsClassifier()
# 训练模型
model.fit(X_train, Y_train)

# 预测
prediction = model.predict(X_test)

score = model.score(X_test, Y_test)
print(score)


print("Clear!")