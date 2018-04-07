import csv
import numpy as np

X = np.zeros((351, 34), dtype='float')
y = np.zeros((351,), dtype='bool')

data_filename = 'ionosphere.data'

with open(data_filename, 'r') as input_file:
    reader = csv.reader(input_file)
    for i, row in enumerate(reader):
        data = [float(datum) for datum in row[:-1]]
        X[i] = data
        y[i] = row[-1] == 'g'

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=14)
print("There are {} samples in the training dataset".format(X_train.shape[0]))
print("There are {} samples in the testing dataset".format(X_test.shape[0]))
print("Each sample has {} features".format(X_train.shape[1]))

from sklearn.neighbors import KNeighborsClassifier

estimator = KNeighborsClassifier()
estimator.fit(X_train, y_train)
y_predicted = estimator.predict(X_test)
accuracy = np.mean(y_test == y_predicted) * 100
print("The accuracy is {0:.3f}%".format(accuracy))

from sklearn.cross_validation import cross_val_score
scores = cross_val_score(estimator, X, y, scoring='accuracy')
average_accuracy = np.mean(scores) * 100
print("The average accuracy is {0:.1f}%".format(average_accuracy))

avg_scores = []
all_scores = []
parameter_values = list(range(1, 21))  
for n_neighbors in parameter_values:
    estimator = KNeighborsClassifier(n_neighbors=n_neighbors)
    scores = cross_val_score(estimator, X, y, scoring='accuracy')
    avg_scores.append(np.mean(scores))
    all_scores.append(scores)

from matplotlib import pyplot as plt
plt.figure(figsize=(32,20))
plt.plot(parameter_values, avg_scores, '-o', linewidth=5, markersize=24)
# plt.axis([0, max(parameter_values), 0, 1.0])


for parameter, scores in zip(parameter_values, all_scores):
    n_scores = len(scores)
    plt.plot([parameter] * n_scores, scores, '-o')

plt.plot(parameter_values, all_scores, 'bx')
plt.show()

# from collections import defaultdict
# all_scores = defaultdict(list)
# parameter_values = list(range(1, 21))  # Including 20
# for n_neighbors in parameter_values:
#     for i in range(100):
#         estimator = KNeighborsClassifier(n_neighbors=n_neighbors)
#         scores = cross_val_score(estimator, X, y, scoring='accuracy', cv=10)
#         all_scores[n_neighbors].append(scores)
# for parameter in parameter_values:
#     scores = all_scores[parameter]
#     n_scores = len(scores)
#     plt.plot([parameter] * n_scores, scores, '-o')

# plt.plot(parameter_values, avg_scores, '-o')