import pandas as pd

data = pd.read_csv('test_data_fm_50.csv')


data = data[data.iloc[:, 0] != 1]

data = data.iloc[:, 1:]

data.columns = ['ID', 'time', 'degradation1', 'degradation2']

# filtered_data = data[~data.index.isin(data.groupby('ID').tail(1).index)]


data.to_csv('processed_test_data.csv', index=False)

