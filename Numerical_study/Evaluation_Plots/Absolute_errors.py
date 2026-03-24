import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# these two units fail early before time 50
EXCLUDED_UNITS_LATER_T = {53, 113}
UNIT_ORDER_ALL = list(range(51, 61)) + list(range(111, 121))

for t in [20, 50, 75]:
    print(f"\n===== t = {t} =====")

    DeepSurv = pd.read_csv(f'../Benchmarks/DeepSurv/Absolute_Error_Results{t}.csv')
    DeepBranched = pd.read_csv(rf"../Benchmarks/DeepBrch/output_t{t}_same_set_paper/AE_values_per_unit.csv")
    Cox = pd.read_csv(f'../Benchmarks/Cox/Absolute_Error_Results{t}.csv')
    ae = pd.read_csv(f'../Prediction/sensor 1_sensor 2_t_{t}/AE_Pivot_{t}.tsv',
                     sep='\t')

    def normalize_unit_column(df, name):
        df = df.copy()

        if "Unit" not in df.columns:
            if len(df) != len(UNIT_ORDER_ALL):
                raise ValueError(
                    f"{name}: cannot assign Unit column automatically. "
                    f"Rows={len(df)}, expected={len(UNIT_ORDER_ALL)}"
                )
            df["Unit"] = UNIT_ORDER_ALL
            print(f"{name}: Unit column added automatically.")
            return df

        unit_vals = pd.to_numeric(df["Unit"], errors="coerce")

        # If Unit is just 1..N row numbering, replace with actual unit ids.
        if len(df) == len(UNIT_ORDER_ALL) and unit_vals.notna().all():
            expected_seq = list(range(1, len(df) + 1))
            if unit_vals.astype(int).tolist() == expected_seq:
                df["Unit"] = UNIT_ORDER_ALL
                print(f"{name}: Unit column replaced from 1..{len(df)} to actual unit ids.")
                return df

        print(f"{name}: Unit column kept as-is.")
        return df

    DeepSurv = normalize_unit_column(DeepSurv, "DeepSurv")
    DeepBranched = normalize_unit_column(DeepBranched, "DeepBranched")
    Cox = normalize_unit_column(Cox, "Cox")
    ae = normalize_unit_column(ae, "ae")

    if t in [50, 75]:
        print(f"Excluding units for t={t}: {sorted(EXCLUDED_UNITS_LATER_T)}")
        DeepSurv = DeepSurv[~DeepSurv["Unit"].isin(EXCLUDED_UNITS_LATER_T)]
        DeepBranched = DeepBranched[~DeepBranched["Unit"].isin(EXCLUDED_UNITS_LATER_T)]
        Cox = Cox[~Cox["Unit"].isin(EXCLUDED_UNITS_LATER_T)]
        ae = ae[~ae["Unit"].isin(EXCLUDED_UNITS_LATER_T)]

    DeepSurv = DeepSurv[["Unit", "Absolute_Error"]].rename(columns={"Absolute_Error": "DeepSurv"})
    DeepBranched = DeepBranched[["Unit", "Absolute_Error"]].rename(columns={"Absolute_Error": "DeepBranched"})
    Cox = Cox[["Unit", "Absolute_Error"]].rename(columns={"Absolute_Error": "Cox"})

    ae = ae.merge(DeepSurv, on="Unit", how="inner")
    ae = ae.merge(DeepBranched, on="Unit", how="inner")
    ae = ae.merge(Cox, on="Unit", how="inner")

    ae = ae.sort_values("Unit").reset_index(drop=True)

    print("Merged units:", ae["Unit"].tolist())

    out_path = f'Absolute_Errors_revision_{t}.csv'
    ae.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")