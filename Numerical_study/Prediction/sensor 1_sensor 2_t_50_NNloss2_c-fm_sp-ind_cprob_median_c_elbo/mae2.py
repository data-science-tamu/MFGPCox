import pandas as pd
import os
import numpy as np
import torch
from itertools import chain

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

t = 50

ae_details = []
coverage_details = []
interval_length_details = []
coverage_interval_ratio_details = []

for unit_name in chain(range(51, 61), range(111, 121)):
    print(f"Processing unit: {unit_name}")

    tsv_file_path = rf"./sp_{t}_exact_ns10000\{unit_name}.tsv"

    try:
        df = pd.read_csv(tsv_file_path, sep="\t")

        times = torch.tensor(df["Time"].values, dtype=torch.float32).to(device)
        true_survival_probs = torch.tensor(df["True_Survival_Probability"].values, dtype=torch.float32).to(device)
        cmgp_probs = torch.tensor(df["Survival_Probability"].values, dtype=torch.float32).to(device)
        cmgp_lower = torch.tensor(df["lower_bound"].values, dtype=torch.float32).to(device)
        cmgp_upper = torch.tensor(df["Upper_bound"].values, dtype=torch.float32).to(device)


        mask = (true_survival_probs > 0) & (cmgp_lower > 0)
        trunc_index = torch.where(mask == False)[0].min().item()-5 if not mask.all() else len(times)




        true_mrl = torch.trapz(true_survival_probs, times).item()
        cmgp_mrl = torch.trapz(cmgp_probs, times).item()
        cmgp_abs_error = abs(true_mrl - cmgp_mrl)


        times = times[:trunc_index]
        true_survival_probs = true_survival_probs[:trunc_index]
        cmgp_probs = cmgp_probs[:trunc_index]
        cmgp_lower = cmgp_lower[:trunc_index]
        cmgp_upper = cmgp_upper[:trunc_index]

        cmgp_coverage = ((true_survival_probs >= cmgp_lower) & (true_survival_probs <= cmgp_upper)).float().mean().item()
        cmgp_interval_length = (cmgp_upper - cmgp_lower).mean().item()
        cmgp_coverage_interval_ratio = cmgp_coverage / cmgp_interval_length if cmgp_interval_length > 0 else 0

        ae_details.append({"Unit": unit_name, "Model": "CMGP-Cox", "AE": cmgp_abs_error})
        coverage_details.append({"Unit": unit_name, "Model": "CMGP-Cox", "Coverage Ratio": cmgp_coverage})
        interval_length_details.append({"Unit": unit_name, "Model": "CMGP-Cox", "Mean Interval Length": cmgp_interval_length})
        coverage_interval_ratio_details.append({"Unit": unit_name, "Model": "CMGP-Cox", "Coverage/Mean Interval Length": cmgp_coverage_interval_ratio})

        for col in df.columns:
            if col.startswith("est_sz_mean_"):
                model_name = col.replace("est_sz_mean_", "")
                model_probs = torch.tensor(df[col].values, dtype=torch.float32).to(device)
                lower_bound_col = col.replace("mean", "lower")
                upper_bound_col = col.replace("mean", "upper")

                if lower_bound_col in df.columns and upper_bound_col in df.columns:
                    model_lower = torch.tensor(df[lower_bound_col].values, dtype=torch.float32).to(device)
                    model_upper = torch.tensor(df[upper_bound_col].values, dtype=torch.float32).to(device)

                    # Truncate additional models
                    model_probs = model_probs[:trunc_index]
                    model_lower = model_lower[:trunc_index]
                    model_upper = model_upper[:trunc_index]

                    model_mrl = torch.trapz(model_probs, times).item()
                    abs_error = abs(true_mrl - model_mrl)
                    model_coverage = ((true_survival_probs >= model_lower) & (true_survival_probs <= model_upper)).float().mean().item()
                    model_interval_length = (model_upper - model_lower).mean().item()
                    model_coverage_interval_ratio = model_coverage / model_interval_length if model_interval_length > 0 else 0

                    ae_details.append({"Unit": unit_name, "Model": model_name, "AE": abs_error})
                    coverage_details.append({"Unit": unit_name, "Model": model_name, "Coverage Ratio": model_coverage})
                    interval_length_details.append({"Unit": unit_name, "Model": model_name, "Mean Interval Length": model_interval_length})
                    coverage_interval_ratio_details.append({"Unit": unit_name, "Model": model_name, "Coverage/Mean Interval Length": model_coverage_interval_ratio})

    except FileNotFoundError:
        print(f"File not found for unit: {unit_name}")
    except Exception as e:
        print(f"Error processing unit {unit_name}: {e}")

ae_details_df = pd.DataFrame(ae_details)
coverage_details_df = pd.DataFrame(coverage_details)
interval_length_details_df = pd.DataFrame(interval_length_details)
coverage_interval_ratio_details_df = pd.DataFrame(coverage_interval_ratio_details)

ranges = {
    "fm1(51,61)": range(51, 62),
    "fm2(111,121)": range(111, 122),
    "all": ae_details_df["Unit"].unique()
}

mae_results = {}
coverage_results = {}
interval_length_results = {}
coverage_interval_ratio_results = {}

for key, unit_range in ranges.items():
    subset_ae = ae_details_df[ae_details_df["Unit"].isin(unit_range)]
    subset_coverage = coverage_details_df[coverage_details_df["Unit"].isin(unit_range)]
    subset_interval_length = interval_length_details_df[interval_length_details_df["Unit"].isin(unit_range)]
    subset_coverage_interval_ratio = coverage_interval_ratio_details_df[coverage_interval_ratio_details_df["Unit"].isin(unit_range)]

    mae_models = {}
    coverage_models = {}
    interval_length_models = {}
    coverage_interval_ratio_models = {}

    for model in subset_ae["Model"].unique():
        mae_models[model] = subset_ae[subset_ae["Model"] == model]["AE"].mean()
        coverage_models[model] = subset_coverage[subset_coverage["Model"] == model]["Coverage Ratio"].mean()
        interval_length_models[model] = subset_interval_length[subset_interval_length["Model"] == model]["Mean Interval Length"].mean()
        coverage_interval_ratio_models[model] = subset_coverage_interval_ratio[subset_coverage_interval_ratio["Model"] == model]["Coverage/Mean Interval Length"].mean()

    mae_results[key] = mae_models
    coverage_results[key] = coverage_models
    interval_length_results[key] = interval_length_models
    coverage_interval_ratio_results[key] = coverage_interval_ratio_models

mae_df = pd.DataFrame(mae_results).T
coverage_df = pd.DataFrame(coverage_results).T
interval_length_df = pd.DataFrame(interval_length_results).T
coverage_interval_ratio_df = pd.DataFrame(coverage_interval_ratio_results).T

mae_output_path = f"MAE_{t}.csv"
coverage_output_path = f"Coverage_{t}.csv"
interval_length_output_path = f"Interval_Length_{t}.csv"
coverage_interval_ratio_output_path = f"Coverage_Interval_Ratio_{t}.csv"

mae_df.to_csv(mae_output_path, sep="\t", index=True)
coverage_df.to_csv(coverage_output_path, sep="\t", index=True)
interval_length_df.to_csv(interval_length_output_path, sep="\t", index=True)
coverage_interval_ratio_df.to_csv(coverage_interval_ratio_output_path, sep="\t", index=True)

ae_pivot_df = ae_details_df.pivot(index="Unit", columns="Model", values="AE")
coverage_pivot_df = coverage_details_df.pivot(index="Unit", columns="Model", values="Coverage Ratio")
interval_length_pivot_df = interval_length_details_df.pivot(index="Unit", columns="Model", values="Mean Interval Length")
coverage_interval_ratio_pivot_df = coverage_interval_ratio_details_df.pivot(index="Unit", columns="Model", values="Coverage/Mean Interval Length")

ae_pivot_output_path = f"AE_Pivot_{t}.tsv"
coverage_pivot_output_path = f"Coverage_Pivot_{t}.tsv"
interval_length_pivot_output_path = f"Interval_Length_Pivot_{t}.tsv"
coverage_interval_ratio_pivot_output_path = f"Coverage_Interval_Ratio_Pivot_{t}.tsv"

ae_pivot_df.to_csv(ae_pivot_output_path, sep="\t")
coverage_pivot_df.to_csv(coverage_pivot_output_path, sep="\t")
interval_length_pivot_df.to_csv(interval_length_pivot_output_path, sep="\t")
coverage_interval_ratio_pivot_df.to_csv(coverage_interval_ratio_pivot_output_path, sep="\t")

