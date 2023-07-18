# data_processing.py
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

# 2変量それぞれで最大値最小値を求め，[0,1]に範囲正規化する
def normalize_fn(df, var1, var2):
    scaler = MinMaxScaler(feature_range=(0, 1))
    return pd.DataFrame(scaler.fit_transform(df[[var1, var2]]), columns=[var1, var2])

# 各データ点と回帰直線との乖離（垂直距離）の平均値を計算する関数を定義
def avg_dist(df, var1, var2):
    X = df[[var1]].values
    y = df[var2].values
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)
    distances = abs(y - y_pred)
    return distances.mean()