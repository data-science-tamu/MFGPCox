

import pandas as pd

for t in [10, 25, 50]:
    # Read datasets
    DeepSurv = pd.read_csv(f'../DeepSurv/Absolute_error_dataset_t{t}.csv')
    # DeepBranched = pd.read_csv(rf"../DeepB/output_t{t}_norm day-time{t}_prob/AE_values_per_unit.csv")
    DeepBranched = pd.read_csv(rf"../DeepB/output_t{t}_same_set_paper/AE_values_per_unit.csv")
    Cox = pd.read_csv(f'../Cox/Absolute_Error_Results{t}.csv')
    NN_J = pd.read_csv(f'../NN_Results/absolute_error_dataset_t{t}.csv')
    # wei = pd.read_csv(f'../MFCMGP-Cox/NL_NH_P13_P26_P3_T3_T6_T42_t_{t}_cprob_weibull_Bayes/AE_Pivot_{t}.tsv', sep='\t')
    ae = pd.read_csv(f'../MFCMGP-Cox/NL_NH_P13_P26_P3_T3_T6_T42_t_{t}_cprob_minV/AE_Pivot_{t}.tsv', sep='\t')
    ae = ae[['Unit', 'CMGP-Cox']].rename(columns={'CMGP-Cox': 'MFCMGP-Cox-exp'})
    # Add Unit column to all datasets
    unit_numbers = pd.Series(range(1, 151))  # Units 1 to 150
    DeepSurv['Unit'] = unit_numbers
    DeepBranched['Unit'] = unit_numbers
    Cox['Unit'] = unit_numbers
    NN_J['Unit'] = unit_numbers
    # wei['Unit'] = unit_numbers

    # Filter each dataset to match the units in ae
    units_in_ae = ae['Unit']
    DeepSurv = DeepSurv[DeepSurv['Unit'].isin(units_in_ae)]
    DeepBranched = DeepBranched[DeepBranched['Unit'].isin(units_in_ae)]
    Cox = Cox[Cox['Unit'].isin(units_in_ae)]
    NN_J = NN_J[NN_J['Unit'].isin(units_in_ae)]


    # Reset indices to match
    DeepSurv = DeepSurv.reset_index(drop=True)
    DeepBranched = DeepBranched.reset_index(drop=True)
    Cox = Cox.reset_index(drop=True)
    NN_J = NN_J.reset_index(drop=True)
    ae = ae.reset_index(drop=True)

    # Add columns from other datasets to ae
    # ae["MFCMGP-Cox-wei"] = wei['CMGP-Cox']
    ae["Deep-B"] = DeepBranched['Absolute_Error']
    ae["NN-J"] = NN_J['absolute_error']
    ae["DeepSurv"] = DeepSurv['absolute_error']
    ae["Cox"] = Cox['Absolute_Error']

    # Save the aligned data
    ae.to_csv(f'Absolute_Errors{t}.csv', index=False)
