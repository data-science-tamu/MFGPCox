import pandas as pd

data = pd.read_csv('all_historical_data.csv')



data = data.iloc[:, 1:]
data = data[data.iloc[:, 0] >= 31]
data = data[data.iloc[:, 0] <= 100]

data.columns = ['ID', 'time'] + [f'degradation{i}' for i in range(1, 9)]

# filtered_data = data[~data.index.isin(data.groupby('ID').tail(1).index)]


data.to_csv('processed_data_fm1_ideal.csv', index=False)

