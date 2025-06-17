import numpy as np
import os
import pandas as pd



for t in [10,25,50, 75]:
    data = {}
    for fm in ['FM1', 'FM2', 'FM3']:
        mrl_values = []
        for unit_idx in range(1, 151):
            file_path = f'../NN_model_{fm}/mrl_loss2_t{t}_ndata150/mrl_{unit_idx}.npy'
            mrl = np.load(file_path)
            mrl_values.append(mrl[0])

        data[fm] = mrl_values

    df = pd.DataFrame(data,index=None)
    df['unit'] = [i for i in range(1, 151)]
    # df.columns = ['unit', 'fm1_RUl', 'fm2_RUl', 'fm3_RUl']
    output_path = f'./mrl_dataset_{t}.csv'
    df.to_csv(output_path)
