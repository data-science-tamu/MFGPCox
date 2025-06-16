import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

for t in [20, 50, 75]:
    DeepSurv = pd.read_csv(f'../Benchmarks/DeepSurv/Absolute_Error_Results{t}.csv')
    # DeepBranched = pd.read_csv(rf"../DeepB/output_t{t}_norm day-time{t}_prob/AE_values_per_unit.csv")
    DeepBranched = pd.read_csv(rf"../DeepBrch/output_t{t}_same_set_paper/AE_values_per_unit.csv")
    Cox = pd.read_csv(f'../Cox/Absolute_Error_Results{t}.csv')
    # point = pd.read_csv(f'../sensor 1_sensor 2_t_{t}_NNloss2_c-fm_sp-ind_cprob_point_var/AE_Pivot_{t}.tsv', sep='\t')
    ae = pd.read_csv(f'../sensor 1_sensor 2_t_{t}_NNloss2_c-fm_sp-ind_cprob_median/AE_Pivot_{t}.tsv', sep='\t')
    ae["DeepSurv"] = DeepSurv['Absolute_Error']
    ae["DeepBranched"] = DeepBranched['Absolute_Error']
    ae["Cox"] = Cox['Absolute_Error']
    # ae["CMGP_Cox_P"] = point['CMGP-Cox']
    ae.to_csv(f'Absolute_Errors{t}.csv', index=False)