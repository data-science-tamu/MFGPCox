import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
import torchtuples as tt
from pycox.models import CoxPH
import gc
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn_pandas import DataFrameMapper
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
import torch
from lifelines import CoxPHFitter
from torch.utils.data import DataLoader, TensorDataset
import csv



# Configuration
np.random.seed(1234)
torch.manual_seed(123)

# Configuration
np.random.seed(1234)
torch.manual_seed(123)


# Utility functions
def load_data(train_path, test_path):
    return pd.read_csv(train_path), pd.read_csv(test_path)


def extract_last_values(data):
    last_values = data.groupby('unit number').last().reset_index()
    last_values['event'] = 1
    last_values.rename(columns={'time, in cycles': 'duration', 'sensor 1': 'sensor1', 'sensor 2': 'sensor2' }, inplace=True)
    columns = ['unit number', 'failure mode', 'duration', 'sensor1', 'sensor2', 'event']
    return last_values[columns]


def prepare_time_varying_data(data):
    data = data.copy()
    data['duration'] = data['time, in cycles']
    data['event'] = 0
    last_cycles = data.groupby('unit number')['duration'].transform('max')
    data.loc[data['duration'] == last_cycles, 'event'] = 1
    return data[['unit number', 'failure mode', 'duration', 'sensor1', 'sensor2', 'event']]


def split_data(df, val_frac=0.1, random_state=123):
    val = df.sample(frac=val_frac, random_state=random_state)
    train = df.drop(val.index)
    return train, val


def preprocess_data(train, val, test, cols_standardize, cols_leave):
    standardize = [([col], StandardScaler()) for col in cols_standardize]
    leave = [(col, None) for col in cols_leave]
    mapper = DataFrameMapper(standardize + leave)

    x_train = mapper.fit_transform(train).astype('float32')
    x_val = mapper.transform(val).astype('float32')
    x_test = mapper.transform(test).astype('float32')

    get_target = lambda df: (df['duration'].values, df['event'].values)
    y_train = get_target(train)
    y_val = get_target(val)
    return x_train, y_train, x_val, y_val, x_test


def train_model_for_exclusion():

    np.random.seed(1234)
    torch.manual_seed(123)


    train_data, test_data = load_data(train_file, test_file)

    # train_data = train_data[train_data.iloc[:, 0] != 2]
    # test_data = test_data[test_data.iloc[:, 0] != 2]

    train_dataset = extract_last_values(train_data)
    test_dataset = extract_last_values(test_data)

    cols_standardize = ['sensor1', 'sensor2']


    cox_model = CoxPHFitter()
    cox_model.fit(train_dataset, duration_col='duration', event_col='event', formula=' + '.join(cols_standardize))
    cox_model.print_summary()



    surv_df = cox_model.predict_survival_function(test_dataset[cols_standardize])

    if cutoff == 50:
        break_index = surv_df.index[surv_df.index > cutoff][0]
        p_t_greater_cutoff = surv_df.loc[break_index]
        conditional_surv = surv_df.loc[break_index + 1:].div(p_t_greater_cutoff, axis=1)
        surv_file_path = f"./surv_sequential{cutoff}.csv"
        conditional_surv.to_csv(surv_file_path, index=True, header=None)

    if cutoff == 75:
        break_index = surv_df.index[surv_df.index > cutoff][0]
        p_t_greater_cutoff = surv_df.loc[break_index]
        conditional_surv = surv_df.loc[break_index + 1:].div(p_t_greater_cutoff, axis=1)
        surv_file_path = f"./surv_sequential{cutoff}.csv"
        conditional_surv = conditional_surv.fillna(0)
        conditional_surv.to_csv(surv_file_path, index=True, header=None)


    else:
        surv_file_path = f"./surv_sequential{cutoff}.csv"
        surv_df.to_csv(surv_file_path, index=True, header=None)

    # Compute integrals
    surv = pd.read_csv(surv_file_path, header=None)
    ts = torch.tensor(surv.iloc[:, 0].values, dtype=torch.float32)
    integral_values = []
    unit_ids = []
    for col in surv.columns[1:]:
        integrand_vals = torch.tensor(surv[col].values, dtype=torch.float32)
        integral_value = torch.trapz(integrand_vals, ts)
        if cutoff == 50:
            integral_values.append(integral_value.item())
            unit_ids.append(col)
            print(f"Unit {col}: Integral = {integral_value.item()}")
        elif cutoff == 75:
            integral_values.append(integral_value.item())
            unit_ids.append(col)
            print(f"Unit {col}: Integral = {integral_value.item()}")
        else:
            integral_values.append(integral_value.item()+ts[0].item()-cutoff)
            unit_ids.append(col)
            print(f"Unit {col}: Integral = {integral_value.item()- (cutoff - ts[0].item())}")

    # Create DataFrame
    df = pd.DataFrame({
        'Unit': unit_ids,
        'RUL': integral_values
    })

    # Clear memory
    del cox_model
    gc.collect()
    torch.cuda.empty_cache()
    return df

for cutoff in [10, 20, 50, 75]:
    train_file = 'historical_data.csv'
    test_file = f'./test_data_fm_{cutoff}.csv'


    results = {}

    df_result = train_model_for_exclusion()
    df_result.to_csv(f'Rul_t{cutoff}.csv', index=False)




time_points = [20, 50, 75]

results = {}

for t in time_points:
    rul = pd.read_csv(f'./Rul_t{t}.csv')

    mrl_data = pd.read_csv(f'./mrl_{t}.csv',header=None)

    mrl_data.columns = ['Actual_RUL']
    mrl_data['Unit'] = rul['Unit']

    merged_data = pd.merge(rul, mrl_data, on='Unit')
    merged_data['Absolute_Error'] = abs(merged_data['RUL'] - merged_data['Actual_RUL'])

    merged_data.to_csv(f'Absolute_Error_Results{t}.csv', index=False)


