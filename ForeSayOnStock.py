import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import median_absolute_error, r2_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import Imputer, PolynomialFeatures

from timeseriesutil import ColumnExtractor, TimeSeriesDiff, TimeSeriesEmbedder

start_time = time.clock()

matplotlib.style.use('ggplot')
plt.xticks(rotation=70)

data = pd.read_csv("aapl-trading-hour.csv",
index_col = 0)

y = data["Close"].diff() / data["Close"].shift()

y[np.isnan(y)]=0

n_total = data.shape[0]
n_train = int(np.ceil(n_total*0.7))

data_train = data[:n_train]
data_test  = data[n_train:]

y_train = y[10:n_train]
y_test  = y[(n_train+10):]

""" 利用Pipeline实现建模
""" 

pipeline = Pipeline([("ColumnEx", ColumnExtractor("Close")),
                     ("Diff", TimeSeriesDiff()),
                     ("Embed", TimeSeriesEmbedder(10)),
                     ("ImputerNA", Imputer()),
                     ("LinReg", LinearRegression())])
                    
pipeline.fit(data_train, y_train)
y_pred = pipeline.predict(data_test)

""" 查看并评价结果
""" 

print(r2_score(y_test, y_pred))
print(median_absolute_error(y_test, y_pred))

cc = np.sign(y_pred)*y_test
cumulative_return = (cc+1).cumprod()
cumulative_return.plot(rot=10)
plt.hold('on') 


pipeline_closing_price = Pipeline([("ColumnEx", ColumnExtractor("Close")),
                                   ("Diff", TimeSeriesDiff()),
                                   ("Embed", TimeSeriesEmbedder(10)),
                                   ("ImputerNA", Imputer())])

pipeline_volume = Pipeline([("ColumnEx", ColumnExtractor("Volume")),
                            ("Diff", TimeSeriesDiff()),
                            ("Embed", TimeSeriesEmbedder(10)),
                            ("ImputerNA", Imputer())])

merged_features = FeatureUnion([("ClosingPriceFeature", pipeline_closing_price),
                                ("VolumeFeature", pipeline_volume)])

pipeline_2 = Pipeline([("MergedFeatures", merged_features),
                       ("PolyFeature",PolynomialFeatures()),
                       ("LinReg", LinearRegression())])
pipeline_2.fit(data_train, y_train)

y_pred_2 = pipeline_2.predict(data_test)

print(r2_score(y_test, y_pred_2))
print(median_absolute_error(y_test, y_pred_2))

cc_2 = np.sign(y_pred_2)*y_test
cumulative_return_2 = (cc_2+1).cumprod()
cumulative_return_2.plot(style="k--", rot=10)
print(time.clock() - start_time, "seconds")
plt.show()

""" 预测运行时间有多长?
""" 

print("Clear!")
