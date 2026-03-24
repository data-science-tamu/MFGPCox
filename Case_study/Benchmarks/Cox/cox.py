# import numpy as np
# import pandas as pd
# import torch
# from sklearn.preprocessing import StandardScaler
# from sklearn_pandas import DataFrameMapper
# import torchtuples as tt
# from pycox.models import CoxPH
# import gc
# import numpy as np
# import pandas as pd
# import torch
# from torch import nn
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# from torch.utils.data import DataLoader, TensorDataset
# from sklearn_pandas import DataFrameMapper
# import gc
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import StandardScaler
# from sklearn_pandas import DataFrameMapper
# import torch
# from lifelines import CoxPHFitter
# from torch.utils.data import DataLoader, TensorDataset
# import csv
#
# # Configuration
# np.random.seed(1234)
# torch.manual_seed(123)
#
# # Configuration
# np.random.seed(1234)
# torch.manual_seed(123)
#
#
#
#
# # Utility functions
# def load_data(train_path, test_path):
#     return pd.read_csv(train_path), pd.read_csv(test_path)
#
#
# def extract_last_values(data):
#     last_values = data.groupby('unit number').last().reset_index()
#     last_values['event'] = 1
#     last_values.rename(columns={'time, in cycles': 'duration'}, inplace=True)
#     columns = ['unit number', 'failure mode', 'duration', "NL", "NH", "P13", "P26", "P3", "T3", "T6", "T42", 'event']
#     return last_values[columns]
#
#
#
# def prepare_time_varying_data(data):
#     data = data.copy()
#     data['duration'] = data['time, in cycles']
#     data['event'] = 0
#     last_cycles = data.groupby('unit number')['duration'].transform('max')
#     data.loc[data['duration'] == last_cycles, 'event'] = 1
#     return data[['unit number', 'failure mode', 'duration', "NL", "NH", "P13", "P26", "P3", "T3", "T6", "T42", 'event']]
#
#
# def split_data(df, val_frac=0.1, random_state=123):
#     val = df.sample(frac=val_frac, random_state=random_state)
#     train = df.drop(val.index)
#     return train, val
#
#
# def preprocess_data(train, val, test, cols_standardize, cols_leave):
#     standardize = [([col], StandardScaler()) for col in cols_standardize]
#     leave = [(col, None) for col in cols_leave]
#     mapper = DataFrameMapper(standardize + leave)
#
#     x_train = mapper.fit_transform(train).astype('float32')
#     x_val = mapper.transform(val).astype('float32')
#     x_test = mapper.transform(test).astype('float32')
#
#     get_target = lambda df: (df['duration'].values, df['event'].values)
#     y_train = get_target(train)
#     y_val = get_target(val)
#     return x_train, y_train, x_val, y_val, x_test
#
#
#
#
# def train_model_for_exclusion():
#
#     np.random.seed(1234)
#     torch.manual_seed(123)
#
#
#     train_data, test_data = load_data(train_file, test_file)
#
#     test_data = test_data[test_data['time, in cycles'] <= cutoff]
#
#     train_dataset = extract_last_values(train_data)
#     test_dataset = extract_last_values(test_data)
#
#
#
#     cols_standardize = ["NL", "NH", "P13", "P26", "P3", "T3", "T6", "T42"]
#
#
#     cox_model = CoxPHFitter()
#     cox_model.fit(train_dataset, duration_col='duration', event_col='event', formula=' + '.join(cols_standardize))
#     cox_model.print_summary()
#
#
#
#     surv_df = cox_model.predict_survival_function(test_dataset[cols_standardize])
#
#     # if cutoff == 50:
#     #     break_index = surv_df.index[surv_df.index > cutoff][0]
#     #     p_t_greater_cutoff = surv_df.loc[break_index]
#     #     conditional_surv = surv_df.loc[break_index + 1:].div(p_t_greater_cutoff, axis=1)
#     #     surv_file_path = f"./surv_sequential{cutoff}.csv"
#     #     conditional_surv.to_csv(surv_file_path, index=True, header=None)
#     #
#     # else:
#     surv_file_path = f"./surv_sequential{cutoff}.csv"
#     surv_df.to_csv(surv_file_path, index=True, header=None)
#
#     # Compute integrals
#     surv = pd.read_csv(surv_file_path, header=None)
#     ts = torch.tensor(surv.iloc[:, 0].values, dtype=torch.float32)
#     integral_values = []
#     unit_ids = []
#     for col in surv.columns[1:]:
#         integrand_vals = torch.tensor(surv[col].values, dtype=torch.float32)
#         integral_value = torch.trapz(integrand_vals, ts)
#         # if cutoff == 50:
#         #     integral_values.append(integral_value.item())
#         #     unit_ids.append(col)
#         #     print(f"Unit {col}: Integral = {integral_value.item()}")
#         # else:
#         integral_values.append(integral_value.item()+ts[0].item()-cutoff)
#         unit_ids.append(col)
#         print(f"Unit {col}: Integral = {integral_value.item()- (cutoff - ts[0].item())}")
#
#     # Create DataFrame
#     df = pd.DataFrame({
#         'Unit': unit_ids,
#         'RUL': integral_values
#     })
#
#     # Clear memory
#     del cox_model
#     gc.collect()
#     torch.cuda.empty_cache()
#     return df
#
# for cutoff in [10, 25, 50, 75]:
#     train_file = './historical_data.csv'
#     test_file = f'./all_historical_data.csv'
#
#
#     results = {}
#
#     df_result = train_model_for_exclusion()
#     df_result.to_csv(f'Rul_t{cutoff}.csv', index=False)
#
#
#
#
#
#
#
# time_points = [10, 25, 50, 75 ]
#
# results = {}
#
# for t in time_points:
#     rul = pd.read_csv(f'./Rul_t{t}.csv')
#
#     mrl_data = pd.read_csv(f'./rul_{t}.csv',header=None)
#
#     mrl_data.columns = ['Actual_RUL']
#     mrl_data['Unit'] = rul['Unit']
#
#     merged_data = pd.merge(rul, mrl_data, on='Unit')
#     merged_data['Absolute_Error'] = abs(merged_data['RUL'] - merged_data['Actual_RUL'])
#
#     merged_data.to_csv(f'Absolute_Error_Results{t}.csv', index=False)
#
#
import os
import gc
import csv
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
from lifelines import CoxPHFitter

# ============================================================
# Configuration
# ============================================================
np.random.seed(1234)
torch.manual_seed(123)

UNIT_COL = 'unit number'
FM_COL = 'failure mode'
TIME_COL = 'time, in cycles'
FEATURE_COLS = ["NL", "NH", "P13", "P26", "P3", "T3", "T6", "T42"]

TRAIN_UNITS = list(range(1, 11)) + list(range(31, 41)) + list(range(101, 111))
ALL_UNITS = list(range(1, 151))
TEST_UNITS = [u for u in ALL_UNITS if u not in TRAIN_UNITS]


# ============================================================
# Utility functions
# ============================================================
def load_full_data(data_path):
    return pd.read_csv(data_path)


def split_train_test_by_unit(all_data, train_units):
    train_data = all_data[all_data[UNIT_COL].isin(train_units)].copy()
    test_data = all_data[~all_data[UNIT_COL].isin(train_units)].copy()
    return train_data, test_data


def extract_last_values(data):
    last_values = data.groupby(UNIT_COL).last().reset_index()
    last_values['event'] = 1
    last_values = last_values.rename(columns={TIME_COL: 'duration'})
    columns = [UNIT_COL, FM_COL, 'duration'] + FEATURE_COLS + ['event']
    return last_values[columns]


def train_model_for_exclusion(train_data, test_data, cutoff):
    np.random.seed(1234)
    torch.manual_seed(123)

    test_data = test_data[test_data[UNIT_COL].isin(TEST_UNITS)].copy()
    test_data = test_data[test_data[TIME_COL] <= cutoff].copy()

    train_dataset = extract_last_values(train_data)
    test_dataset = extract_last_values(test_data)

    test_dataset = test_dataset[test_dataset[UNIT_COL].isin(TEST_UNITS)].copy()
    test_dataset = test_dataset.sort_values(UNIT_COL).reset_index(drop=True)

    cox_model = CoxPHFitter()
    cox_model.fit(
        train_dataset,
        duration_col='duration',
        event_col='event',
        formula=' + '.join(FEATURE_COLS)
    )

    surv_df = cox_model.predict_survival_function(test_dataset[FEATURE_COLS])

    surv_file_path = f"./surv_sequential{cutoff}.csv"
    surv_df.to_csv(surv_file_path, index=True, header=None)

    surv = pd.read_csv(surv_file_path, header=None)
    ts = torch.tensor(surv.iloc[:, 0].values, dtype=torch.float32)

    integral_values = []
    unit_ids = test_dataset[UNIT_COL].tolist()

    for idx, col in enumerate(surv.columns[1:]):
        integrand_vals = torch.tensor(surv[col].values, dtype=torch.float32)
        integral_value = torch.trapz(integrand_vals, ts)

        rul_value = integral_value.item() + ts[0].item() - cutoff
        integral_values.append(rul_value)

        print(f"Unit {unit_ids[idx]}: Integral = {rul_value}")

    df = pd.DataFrame({
        'Unit': unit_ids,
        'RUL': integral_values
    })

    df = df[df['Unit'].isin(TEST_UNITS)].copy()
    df = df.sort_values('Unit').reset_index(drop=True)

    del cox_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return df


# ============================================================
# Load data and split by unit
# ============================================================
all_data = load_full_data('./all_historical_data.csv')
train_data, test_data = split_train_test_by_unit(all_data, TRAIN_UNITS)

train_data = train_data[train_data[UNIT_COL].isin(TRAIN_UNITS)].copy()
test_data = test_data[test_data[UNIT_COL].isin(TEST_UNITS)].copy()

print("Train units:", sorted(train_data[UNIT_COL].unique()))
print("Number of train units:", len(train_data[UNIT_COL].unique()))
print("Number of test units:", len(test_data[UNIT_COL].unique()))


# ============================================================
# Train and save RUL predictions for test units only
# ============================================================
for cutoff in [10, 25, 50, 75]:
    df_result = train_model_for_exclusion(train_data, test_data, cutoff)
    df_result.to_csv(f'Rul_t{cutoff}.csv', index=False)


# ============================================================
# Absolute error calculation for test units only
# ============================================================
time_points = [10, 25, 50, 75]

for t in time_points:
    rul = pd.read_csv(f'./Rul_t{t}.csv')
    rul = rul[rul['Unit'].isin(TEST_UNITS)].copy()
    rul = rul.sort_values('Unit').reset_index(drop=True)

    mrl_data = pd.read_csv(f'./rul_{t}.csv', header=None)
    mrl_data.columns = ['Actual_RUL']

    # If rul_t.csv contains all 150 units in order, attach 1..150 first
    mrl_data['Unit'] = list(range(1, len(mrl_data) + 1))

    # Then keep only the true test units
    mrl_data = mrl_data[mrl_data['Unit'].isin(TEST_UNITS)].copy()
    mrl_data = mrl_data.sort_values('Unit').reset_index(drop=True)

    merged_data = pd.merge(rul, mrl_data, on='Unit', how='inner')
    merged_data = merged_data[merged_data['Unit'].isin(TEST_UNITS)].copy()
    merged_data['Absolute_Error'] = abs(merged_data['RUL'] - merged_data['Actual_RUL'])
    merged_data = merged_data.sort_values('Unit').reset_index(drop=True)

    merged_data.to_csv(f'Absolute_Error_Results{t}.csv', index=False)