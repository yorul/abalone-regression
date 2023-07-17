# data_processing.py
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def normalize_fn(df, var1, var2):
    scaler = MinMaxScaler(feature_range=(0, 1))
    return pd.DataFrame(scaler.fit_transform(df[[var1, var2]]), columns=[var1, var2])
