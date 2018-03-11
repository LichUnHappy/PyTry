import pandas as pd 
from datetime import datetime


data = pd.read_csv("aapl.csv",
                   names = ["timestamp_raw","Open","High",
                            "Low","Close","Volume"],
                    index_col = False)

print(type(data))
print(data.head(5))
print(data.tail(5))

UNIX_EPOCH = datetime(1970, 1, 1, 0, 0)

def ConvertTime(timestamp_raw, data):
        """该函数会将原始时间转化为所需要的datatime格式"""
        delta = datetime.utcfromtimestamp(timestamp_raw) - UNIX_EPOCH
        return  data + delta
data.index = map(lambda x: ConvertTime(x, datetime(2015, 8, 3)),
                data["timestamp_raw"] / 1000)

data = data.drop("timestamp_raw", 1)

print(data.head(5))
print(data.tail(5))

print(data.describe())
print(data.index.min())
print(data.index.max())

data_trading_hour = data["201508030930":"201508031529"]

import matplotlib
from matplotlib import pyplot as plt

matplotlib.style.use('ggplot')
# data_trading_hour["Close"].plot()
# plt.show()

# data_trading_hour["Volume"].plot.hist()
# plt.show()

# data_trading_hour["Volume"].describe()
# data_trading_hour["Volume"].plot()
# plt.show()

# data_trading_hour["High"].describe()
# data_trading_hour["High"].plot()
# plt.show()

# data_trading_hour["Close"].diff().plot.hist()
# plt.show()

change = data_trading_hour["Close"].diff() / data_trading_hour["Close"]
# change.plot()
# plt.show()

change.shift(1).corr(change)
change.shift(2).corr(change)

plt.acorr(change[1:], lw = 2)
plt.show()

print("Clear!")