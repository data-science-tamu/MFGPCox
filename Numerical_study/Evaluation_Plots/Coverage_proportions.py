import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# these two units fail early before time 50
EXCLUDED_UNITS_LATER_T = {53, 113}
ALL_UNITS = list(range(51, 61)) + list(range(111, 121))

for t in [20, 50, 75]:
    ae = pd.read_csv(
        f'../Prediction/sensor 1_sensor 2_t_{t}/Coverage_Pivot_{t}.tsv',
        sep='\t'
    )

    if "Unit" not in ae.columns:
        if len(ae) != len(ALL_UNITS):
            raise ValueError(
                f"Cannot assign Unit automatically for t={t}: "
                f"rows={len(ae)}, expected={len(ALL_UNITS)}"
            )
        ae = ae.copy()
        ae["Unit"] = ALL_UNITS
    else:
        unit_vals = pd.to_numeric(ae["Unit"], errors="coerce")
        expected_seq = list(range(1, len(ae) + 1))
        if len(ae) == len(ALL_UNITS) and unit_vals.notna().all():
            if unit_vals.astype(int).tolist() == expected_seq:
                ae = ae.copy()
                ae["Unit"] = ALL_UNITS

    if t in [50, 75]:
        ae = ae[~ae["Unit"].isin(EXCLUDED_UNITS_LATER_T)].reset_index(drop=True)

    ae.to_csv(f'Coverage{t}.csv', index=False)
