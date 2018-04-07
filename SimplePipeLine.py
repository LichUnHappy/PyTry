import csv
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score

X = np.zeros((351, 34), dtype='float')
y = np.zeros((351,), dtype='bool')

data_filename = 'ionosphere.data'

with open(data_filename, 'r') as input_file:
    reader = csv.reader(input_file)
    for i, row in enumerate(reader):
        data = [float(datum) for datum in row[:-1]]
        X[i] = data
        y[i] = row[-1] == 'g'

X_broken = np.array(X)
X_broken[:,::2]  /= 10


# X_transformed = MinMaxScaler().fit_transform(X_broken)
# estimator = KNeighborsClassifier()
# transformed_scores = cross_val_score(estimator, X_transformed, y, scoring='accuracy')
# print("The average accuracy for is {0:.1f}%".format(np.mean(transformed_scores * 100)))


from sklearn.pipeline import Pipeline

scaling_pipeline = Pipeline([('scale', MinMaxScaler()),
                            ('predict', KNeighborsClassifier())])

scores = cross_val_score(scaling_pipeline, X_broken, y, scoring='accuracy')
print("The pipeline score is {0:.1f}%".format(np.mean(scores) * 100))