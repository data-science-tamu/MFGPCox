

import pandas as pd

for t in [10, 25,50, 75]:
    # Read datasets
    DeepSurv = pd.read_csv(f'../DeepSurv/save/t_{t}/all_units_probabilities.tsv', sep='\t')
    DeepSurv.columns = ['unit', 'DeepSurv_p1', 'DeepSurv_p2', 'DeepSurv_p3']

    DeepBranched = pd.read_csv(rf"../DeepBrch/output_t{t}_same_set_paper/classifier_result.csv")
    DeepBranched = DeepBranched.drop(columns=DeepBranched.columns[0])
    DeepBranched.columns = ['DeepBranched_p1', 'DeepBranched_p2', 'DeepBranched_p3']

    NN_joint = pd.read_csv(f'../NN_model_prob/fm_probabilities/t_{t}/fm_probabilities_t_{t}.csv')
    NN_joint = NN_joint.drop(columns=NN_joint.columns[0])
    NN_joint.columns = ['NN_joint_p1', 'NN_joint_p2', 'NN_joint_p3']

    cmgp = pd.read_csv(f'../MFCMGP-Cox/NL_NH_P13_P26_P3_T3_T6_T42_t_{t}_cprob_minV/failure_mode_probs_{t}.csv')
    cmgp = cmgp.drop(columns=cmgp.columns[4])
    cmgp.columns = ['Units', 'CMGP_Cox_p1', 'CMGP_Cox_p2', 'CMGP_Cox_p3', 'Actual_Failure_Mode']

    # Filter datasets based on cmgp['Units']
    cmgp_units = cmgp['Units']
    DeepSurv = DeepSurv[DeepSurv['unit'].isin(cmgp_units)].reset_index(drop=True)
    DeepBranched['unit'] = range(1, 151)  # Add unit column if missing
    DeepBranched = DeepBranched[DeepBranched['unit'].isin(cmgp_units)].reset_index(drop=True)
    NN_joint['unit'] = range(1, 151)  # Add unit column if missing
    NN_joint = NN_joint[NN_joint['unit'].isin(cmgp_units)].reset_index(drop=True)

    # Concatenate datasets
    concatenated_rows = pd.concat([cmgp.reset_index(drop=True),
                                   DeepSurv.drop(columns=['unit']),
                                   DeepBranched.drop(columns=['unit']),
                                   NN_joint.drop(columns=['unit'])], axis=1)

    # Save the result
    concatenated_rows.to_csv(f'classes{t}.csv', index=False)
