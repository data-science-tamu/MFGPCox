import pandas as pd
import os
import numpy as np
import torch
from itertools import chain

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

t = 10

ae_details = []

for unit_name in chain(range(11, 31), range(41, 101), range(111, 151)):
    print(f"Processing unit: {unit_name}")

    tsv_file_path = rf"./sp_{t}_exact_ns10000/{unit_name}.tsv"

    try:
        df = pd.read_csv(tsv_file_path, sep="\t")

        times = torch.tensor(df["Time"].values, dtype=torch.float32).to(device)
        true_mrl = torch.tensor(df["Actual_Failure"].values, dtype=torch.float32).to(device)
        true_mrl = torch.unique(true_mrl).item() - t
        cmgp_probs = torch.tensor(df["Survival_Probability"].values, dtype=torch.float32).to(device)
        cmgp_lower = torch.tensor(df["lower_bound"].values, dtype=torch.float32).to(device)
        cmgp_upper = torch.tensor(df["Upper_bound"].values, dtype=torch.float32).to(device)


        cmgp_mrl = torch.trapz(cmgp_probs, times).item()
        cmgp_abs_error = abs(true_mrl - cmgp_mrl)


        ae_details.append({
            "Unit": unit_name,
            "Model": "CMGP-Cox",
            "True_RUL": true_mrl,
            "Predicted_RUL": cmgp_mrl,
            "AE": cmgp_abs_error

        })


    except FileNotFoundError:
        print(f"File not found for unit: {unit_name}")
    except Exception as e:
        print(f"Error processing unit {unit_name}: {e}")

ae_details_df = pd.DataFrame(ae_details)


ranges = {
    "fm1(11,30)": range(11, 31),
    "fm2(41,100)": range(41, 401),
    "fm3(101,150)": range(111, 151),
    "all": ae_details_df["Unit"].unique()
}

mae_results = {}
coverage_results = {}
interval_length_results = {}
coverage_interval_ratio_results = {}

for key, unit_range in ranges.items():
    subset_ae = ae_details_df[ae_details_df["Unit"].isin(unit_range)]


    mae_models = {}
    coverage_models = {}
    interval_length_models = {}
    coverage_interval_ratio_models = {}

    for model in subset_ae["Model"].unique():
        mae_models[model] = subset_ae[subset_ae["Model"] == model]["AE"].mean()


    mae_results[key] = mae_models


mae_df = pd.DataFrame(mae_results).T


mae_output_path = f"MAE_{t}.csv"


mae_df.to_csv(mae_output_path, sep="\t", index=True)


ae_pivot_df = ae_details_df.pivot(index="Unit", columns="Model", values="AE")

ae_pivot_output_path = f"AE_Pivot_{t}.tsv"

ae_pivot_df.to_csv(ae_pivot_output_path, sep="\t")


