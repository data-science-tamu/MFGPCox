import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

for t in [20, 50, 75]:
    DeepSurv = pd.read_csv(f'../DeepSurv/save/t_{t}/all_units_probabilities.tsv', sep='\t')
    DeepSurv.columns = ['unit', 'DeepSurv_p1', 'DeepSurv_p2']

    DeepBranched = pd.read_csv(rf"../DeepB/output_t{t}_norm day-time{t}_prob/classifier_result.csv")
    DeepBranched = DeepBranched.drop(columns=DeepBranched.columns[0])
    DeepBranched.columns = ['DeepBranched_p1', 'DeepBranched_p2']

    NN_joint_ideal = pd.read_csv(f'../NN_model_NS_19_ideal_prob_2/fm_probabilities/t_{t}/fm_probabilities_t_{t}.csv')
    NN_joint_ideal = NN_joint_ideal.drop(columns=NN_joint_ideal.columns[0])
    NN_joint_ideal.columns = ['NN_joint_ideal_p1', 'NN_joint_ideal_p2']

    NN_joint_mis = pd.read_csv(f'../NN_model_NS_19_ideal_prob_2/fm_probabilities/t_{t}/fm_probabilities_t_{t}.csv')
    NN_joint_mis = NN_joint_mis.drop(columns=NN_joint_mis.columns[0])
    NN_joint_mis.columns = ['NN_joint_mis_p1', 'NN_joint_mis_p2']

    cmgp = pd.read_csv(f'../sensor 1_sensor 2_t_{t}_NNloss2_c-fm_sp-ind_cprob_median_c_elbo/failure_mode_probs_{t}.csv')
    cmgp = cmgp.drop(columns=cmgp.columns[3])
    cmgp = cmgp.drop(columns=cmgp.columns[0])
    cmgp.columns = ['CMGP_Cox_p1', 'CMGP_Cox_p2', 'Actual_Failure_Mode']

    concatenated_rows = pd.concat([DeepSurv, DeepBranched, NN_joint_ideal, NN_joint_mis, cmgp], axis=1)

    concatenated_rows.to_csv(f'classes{t}.csv', index=False)
