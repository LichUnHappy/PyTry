import pandas as pd 
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

"""
fit-transform <- prepocessing
""" 

def embed_time_series(x, k):
    """this function would transform an N dimensional time series into a
    tuple containing: 
    1) an (n - k) by k matrix that is [X[i], x[i+1], ... x[i+k-1]],
    for i from 0 to n-k-1
    
    2) a vector of length (n - k) that is [x[k], x[k+1] ... x[n]]
    """
    n = len(x)

    if k >= n: 
        raise "Can not deal with k greater than the length of x" 
    
    output_x = list(map(lambda i: list(x[i:(i+k)]), 
                        range(0, n-k)))
    return np.array(output_x)

class TimeSeriesEmbedder(BaseEstimator, TransformerMixin):
    def __init__(self, k):
        self.k = k 
    def fit(self, X, y= None):
        return self
    def transform(self, X, y = None):
        return embed_time_series(X, self.k)

class ColumnExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, column_name):
        self.column_name = column_name
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return X[self.column_name]

class TimeSeriesDiff(BaseEstimator, TransformerMixin):
    def __init__(self, k=1):
        self.k = k 
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        if type(X) is pd.core.frame.DataFrame or type(X) is pd.core.series.Series:
            return X.diff(self.k) / X.shift(self.k)
        else:
            raise "Have to be a pandas data frame or Series object!"
