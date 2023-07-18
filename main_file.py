import pandas as pd
from itertools import combinations
from sklearn.preprocessing import MinMaxScaler
from data_loading import load_data
from data_processing import normalize_fn, avg_dist

df, columns = load_data("./abalone.data")

# 8種より2種の選択の組み合わせを全て示す
features = columns[1:-1]
feature_combinations = list(combinations(features, 2))

# 雌雄幼体のクラス別にデータを分けて、すべての特徴量の組み合わせについてデータの正規化と線形回帰を行う
results = []
for sex in ['M', 'F', 'I']:
    data_sex = df[df['Sex'] == sex]
    for var1, var2 in feature_combinations:
        # データを正規化する
        data_normalized = normalize_fn(data_sex, var1, var2)

        # 線形回帰を行い、平均距離を計算する
        average_distance = avg_dist(data_normalized, var1, var2)
        results.append((sex, var1, var2, average_distance))

# 結果をデータフレームに変換し、平均距離でソートする
results_df = pd.DataFrame(results, columns=['性別', '特徴量１', '特徴量２', '乖離値'])
results_df = results_df.sort_values(by='乖離値')

# 最も乖離の平均値が小さい5つの組み合わせを取得する
best_results = results_df.groupby('性別').head(5)

print(best_results)



