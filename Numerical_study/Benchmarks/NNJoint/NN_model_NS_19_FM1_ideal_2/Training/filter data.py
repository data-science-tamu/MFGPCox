import pandas as pd

data = pd.read_csv('all_units_time_series_fm1.csv')



data = data.iloc[:, 1:]

data = data[data.iloc[:, 0] <= 50]

data.columns = ['ID', 'time', 'degradation1', 'degradation2']

# filtered_data = data[~data.index.isin(data.groupby('ID').tail(1).index)]


data.to_csv('processed_data_fm1_ideal.csv', index=False)

