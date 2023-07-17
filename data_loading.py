# data_loading.py
import pandas as pd

def load_data(filepath):
    # データを読み込みます
    columns = ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight',
                'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings']
    df = pd.read_csv(filepath, names=columns)
    return df, columns

