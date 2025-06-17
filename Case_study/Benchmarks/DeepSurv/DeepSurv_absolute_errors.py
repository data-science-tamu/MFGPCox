import pandas as pd

# probabilities_path = 'path/to/all_units_probabilities.tsv'

time_points = [10, 25, 50, 75]
rul_files = {
    10: {'fm1': './FM1_Rul_t10.csv', 'fm2': './FM2_Rul_t10.csv', 'fm3': './FM3_Rul_t10.csv'},
    25: {'fm1': './FM1_Rul_t25.csv', 'fm2': './FM2_Rul_t25.csv', 'fm3': './FM3_Rul_t25.csv'},
    50: {'fm1': './FM1_Rul_t50.csv', 'fm2': './FM2_Rul_t50.csv', 'fm3': './FM3_Rul_t50.csv'},
    75: {'fm1': './FM1_Rul_t75.csv', 'fm2': './FM2_Rul_t75.csv', 'fm3': './FM3_Rul_t75.csv'},
}


results = {}

for t in time_points:
    fm1_rul = pd.read_csv(rul_files[t]['fm1'])
    fm2_rul = pd.read_csv(rul_files[t]['fm2'])
    fm3_rul = pd.read_csv(rul_files[t]['fm3'])
    probabilities = pd.read_csv(f'./save/t_{t}/all_units_probabilities.tsv', sep='\t')

    merged = (
        probabilities.merge(fm1_rul, left_on='ID', right_on='Unit', how='inner')
        .merge(fm2_rul, left_on='ID', right_on='Unit', how='inner', suffixes=('_fm1', '_fm2'))
        .merge(fm3_rul, left_on='ID', right_on='Unit', how='inner', suffixes=('', '_fm3'))
    )

    merged['Weighted_RUL'] = merged['fm_1_prob'] * merged['RUL_fm1'] + merged['fm_2_prob'] * merged['RUL_fm2'] + merged['fm_3_prob'] * merged['RUL']
    results[t] = merged[['ID', 'Weighted_RUL']]
    output_path = f'Weighted_RUL_t{t}.csv'
    results[t].to_csv(output_path, index=False)
    print(f"Weighted RUL for t={t} has been calculated and saved to '{output_path}'.")

    weighted_rul_data = pd.read_csv(f'./Weighted_RUL_t{t}.csv')

    historical_data_path = '../../all_historical_data.csv'
    historical_data = pd.read_csv(historical_data_path)

    historical_data = historical_data.rename(columns={'unit number': 'ID', 'time, in cycles': 'time'})
    final_data = historical_data.merge(weighted_rul_data, on='ID')

    final_data['failure_time'] = final_data.groupby('ID')['time'].transform('max')
    final_data['true_rul'] = final_data['failure_time'] - t

    final_data['absolute_error'] = abs(final_data['Weighted_RUL'] - final_data['true_rul'])

    result_data = final_data[['ID', 'true_rul', 'Weighted_RUL', 'absolute_error']].drop_duplicates()

    RUL_data = final_data[['ID','true_rul']].drop_duplicates(subset='ID')
    RUL_data = RUL_data['true_rul']


    result_data.to_csv(f'Absolute_error_dataset_t{t}.csv', index=False)
    RUL_data.to_csv(f'rul_{t}.csv', index=False, header=None)








