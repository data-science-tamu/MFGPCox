# import numpy as np
# import os
# import pandas as pd
#
#
# # mrl = np.load(r"C:\Users\sina.aghaee\Desktop\GitLab\condition-monitoring\CS-8s\final\NN_model_FM3\mrl_loss2_t10_ndata150\mrl_3.npy")
# # print()
#
#
# for t in [10,25,50, 75]:
#     data = {}
#     for fm in ['FM1', 'FM2', 'FM3']:
#         mrl_values = []
#         for unit_idx in range(1, 151):
#             file_path = f'../NN_model_{fm}/mrl_loss2_t{t}_ndata150/mrl_{unit_idx}.npy'
#             mrl = np.load(file_path)
#             mrl_values.append(mrl[0])
#
#         data[fm] = mrl_values
#
#     df = pd.DataFrame(data,index=None)
#     df['unit'] = [i for i in range(1, 151)]
#     # df.columns = ['unit', 'fm1_RUl', 'fm2_RUl', 'fm3_RUl']
#     output_path = f'./mrl_dataset_{t}.csv'
#     df.to_csv(output_path)
#

import numpy as np
import os
import pandas as pd

# Define train units
TRAIN_UNITS = list(range(1, 11)) + list(range(31, 41)) + list(range(101, 111))

# Define all units (1 to 150)
ALL_UNITS = list(range(1, 151))

# Test units = everything except train units
TEST_UNITS = [u for u in ALL_UNITS if u not in TRAIN_UNITS]


for t in [10, 25, 50, 75]:
    data = {}

    for fm in ['FM1', 'FM2', 'FM3']:
        mrl_values = []

        for unit_idx in TEST_UNITS:
            file_path = f'../NN_model_{fm}/mrl_loss2_t{t}_ndata150/mrl_{unit_idx}.npy'
            mrl = np.load(file_path)
            mrl_values.append(mrl[0])

        data[fm] = mrl_values

    # Create DataFrame ONLY for test units
    df = pd.DataFrame(data)
    df['unit'] = TEST_UNITS

    # Optional: reorder columns
    df = df[['unit', 'FM1', 'FM2', 'FM3']]

    output_path = f'./mrl_dataset_{t}.csv'
    df.to_csv(output_path, index=False)