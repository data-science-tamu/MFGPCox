import pandas as pd

# probabilities_path = 'path/to/all_units_probabilities.tsv'

time_points = [20, 50, 75]
rul_files = {
    10: {'fm1': './FM1_Rul_t10.csv', 'fm2': './FM2_Rul_t10.csv'},
    20: {'fm1': './FM1_Rul_t20.csv', 'fm2': './FM2_Rul_t20.csv'},
    50: {'fm1': './FM1_Rul_t50.csv', 'fm2': './FM2_Rul_t50.csv'},
    75: {'fm1': './FM1_Rul_t75.csv', 'fm2': './FM2_Rul_t75.csv'}
}


results = {}

for t in time_points:
    fm1_rul = pd.read_csv(rul_files[t]['fm1'])
    fm2_rul = pd.read_csv(rul_files[t]['fm2'])
    probabilities = pd.read_csv(f'./save/t_{t}/all_units_probabilities.tsv', sep='\t')
    fm1_rul.loc[:9, 'Unit'] = range(51, 61)
    fm1_rul.loc[10:19, 'Unit'] = range(111, 121)
    fm2_rul.loc[:9, 'Unit'] = range(51, 61)
    fm2_rul.loc[10:19, 'Unit'] = range(111, 121)
    merged = probabilities.merge(fm1_rul, left_on='ID', right_on='Unit', how='inner')
    merged = merged.merge(fm2_rul, left_on='ID', right_on='Unit', how='inner', suffixes=('_fm1', '_fm2'))
    merged['Weighted_RUL'] = merged['fm_1_prob'] * merged['RUL_fm1'] + merged['fm_2_prob'] * merged['RUL_fm2']
    results[t] = merged[['ID', 'Weighted_RUL']]
    output_path = f'Weighted_RUL_t{t}.csv'
    results[t].to_csv(output_path, index=False)
    print(f"Weighted RUL for t={t} has been calculated and saved to '{output_path}'.")

    weighted_rul_data = pd.read_csv(f'./Weighted_RUL_t{t}.csv')
    mrl_data = pd.read_csv(f'./mrl_{t}.csv',header=None)

    mrl_data.columns = ['Actual_RUL']
    mrl_data['ID'] = weighted_rul_data['ID']

    merged_data = pd.merge(weighted_rul_data, mrl_data, on='ID')
    merged_data['Absolute_Error'] = abs(merged_data['Weighted_RUL'] - merged_data['Actual_RUL'])

    merged_data.to_csv(f'Absolute_Error_Results{t}.csv', index=False)






