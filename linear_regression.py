# linear_regression.py
from sklearn.linear_model import LinearRegression

def train_and_predict(X, y):
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)
    return y_pred

def calculate_average_distance(y, y_pred):
    distances = abs(y - y_pred)
    return distances.mean()
