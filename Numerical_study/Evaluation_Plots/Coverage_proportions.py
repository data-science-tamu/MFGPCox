import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

for t in [20, 50, 75]:
    ae = pd.read_csv(f'../Prediction/sensor 1_sensor 2_t_{t}_NNloss2_c-fm_sp-ind_cprob_median_c_elbo/Coverage_Pivot_{t}.tsv', sep='\t')

    ae.to_csv(f'Coverage{t}.csv', index=False)
