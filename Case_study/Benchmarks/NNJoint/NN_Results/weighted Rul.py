# import pandas as pd
#
#
#
#
# for t in [10, 25,50,75]:
#     mrl_file_path = f'./mrl_dataset_{t}.csv'
#     probabilities_file_path = f'../NN_model_prob/fm_probabilities/t_{t}/fm_probabilities_t_{t}.csv'
#
#
#     mrl_data = pd.read_csv(mrl_file_path)
#     fm_probabilities = pd.read_csv(probabilities_file_path)
#
#     mrl_data = mrl_data.drop(columns=['Unnamed: 0'])
#
#     merged_data = mrl_data.merge(fm_probabilities, left_on='unit', right_on='ID')
#
#     merged_data['weighted_rul'] = (
#         merged_data['FM1'] * merged_data['fm_1_prob'] +
#         merged_data['FM2'] * merged_data['fm_2_prob'] +
#         merged_data['FM3'] * merged_data['fm_3_prob']
#     )
#
#     merged_data.to_csv(f'weighted_rul{t}.csv', index=False)
#
#
#
#
#
#
#     historical_data_path = './all_historical_data.csv'
#     historical_data = pd.read_csv(historical_data_path)
#
#     historical_data = historical_data.rename(columns={'unit number': 'unit', 'time, in cycles': 'time'})
#     final_data = historical_data.merge(merged_data, on='unit')
#
#     final_data['failure_time'] = final_data.groupby('unit')['time'].transform('max')
#     final_data['true_rul'] = final_data['failure_time'] - t
#
#     final_data['absolute_error'] = abs(final_data['weighted_rul'] - final_data['true_rul'])
#
#     result_data = final_data[['unit', 'true_rul', 'weighted_rul', 'absolute_error']].drop_duplicates()
#
#     result_data.to_csv(f'absolute_error_dataset_t{t}.csv', index=False)


import pandas as pd

# Define train units
TRAIN_UNITS = list(range(1, 11)) + list(range(31, 41)) + list(range(101, 111))

# Define all units
ALL_UNITS = list(range(1, 151))

# Test units only
TEST_UNITS = [u for u in ALL_UNITS if u not in TRAIN_UNITS]


for t in [10, 25, 50, 75]:
    mrl_file_path = f'./mrl_dataset_{t}.csv'
    probabilities_file_path = f'../NN_model_prob/fm_probabilities/t_{t}/fm_probabilities_t_{t}.csv'

    mrl_data = pd.read_csv(mrl_file_path)
    fm_probabilities = pd.read_csv(probabilities_file_path)

    # Drop unwanted unnamed column only if it exists
    if 'Unnamed: 0' in mrl_data.columns:
        mrl_data = mrl_data.drop(columns=['Unnamed: 0'])

    # Keep only test units in both files
    mrl_data = mrl_data[mrl_data['unit'].isin(TEST_UNITS)].copy()
    fm_probabilities = fm_probabilities[fm_probabilities['ID'].isin(TEST_UNITS)].copy()

    merged_data = mrl_data.merge(fm_probabilities, left_on='unit', right_on='ID', how='inner')

    merged_data['weighted_rul'] = (
        merged_data['FM1'] * merged_data['fm_1_prob'] +
        merged_data['FM2'] * merged_data['fm_2_prob'] +
        merged_data['FM3'] * merged_data['fm_3_prob']
    )

    merged_data.to_csv(f'weighted_rul{t}.csv', index=False)

    historical_data_path = './all_historical_data.csv'
    historical_data = pd.read_csv(historical_data_path)

    historical_data = historical_data.rename(columns={'unit number': 'unit', 'time, in cycles': 'time'})

    # Keep only test units in historical data too
    historical_data = historical_data[historical_data['unit'].isin(TEST_UNITS)].copy()

    final_data = historical_data.merge(merged_data, on='unit', how='inner')

    final_data['failure_time'] = final_data.groupby('unit')['time'].transform('max')
    final_data['true_rul'] = final_data['failure_time'] - t
    final_data['absolute_error'] = abs(final_data['weighted_rul'] - final_data['true_rul'])

    result_data = final_data[['unit', 'true_rul', 'weighted_rul', 'absolute_error']].drop_duplicates()
    result_data = result_data.sort_values('unit').reset_index(drop=True)

    result_data.to_csv(f'absolute_error_dataset_t{t}.csv', index=False)