import pandas as pd
import numpy as np
import torch
from itertools import chain

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

t = 20

# Load failure times and compute true RUL
file_path_2 = "./rul-my.csv"
failure_time = pd.read_csv(file_path_2, header=None)

unit_list = list(range(51, 61)) + list(range(111, 121))
true_rul_dict = dict(zip(unit_list, (failure_time[0] - t).values))

ae_details = []
coverage_details = []
interval_length_details = []
coverage_interval_ratio_details = []

for unit_name in unit_list:
    print(f"Processing unit: {unit_name}")

    tsv_file_path = rf"C:\Users\sina.aghaee\Desktop\GitLab\CM\numerical_study\numerical_study_19\sensor 1_sensor 2_t_{t}_NNloss2_c-fm_sp-ind_cprob_median_c_elbo\sp_{t}_exact_ns10000\{unit_name}.tsv"

    try:
        df = pd.read_csv(tsv_file_path, sep="\t")

        times = torch.tensor(df["Time"].values, dtype=torch.float32).to(device)
        true_survival_probs = torch.tensor(df["True_Survival_Probability"].values, dtype=torch.float32).to(device)
        cmgp_probs = torch.tensor(df["Survival_Probability"].values, dtype=torch.float32).to(device)
        cmgp_lower = torch.tensor(df["lower_bound"].values, dtype=torch.float32).to(device)
        cmgp_upper = torch.tensor(df["Upper_bound"].values, dtype=torch.float32).to(device)

        mask = (true_survival_probs > 0) & (cmgp_lower > 0)
        trunc_index = torch.where(mask == False)[0].min().item() if not mask.all() else len(times)

        true_rul_unit = true_rul_dict[unit_name]
        cmgp_mrl = torch.trapz(cmgp_probs[:trunc_index], times[:trunc_index]).item()
        cmgp_abs_error = abs(true_rul_unit - cmgp_mrl)

        cmgp_coverage = ((true_survival_probs[:trunc_index] >= cmgp_lower[:trunc_index]) &
                         (true_survival_probs[:trunc_index] <= cmgp_upper[:trunc_index])).float().mean().item()
        cmgp_interval_length = (cmgp_upper[:trunc_index] - cmgp_lower[:trunc_index]).mean().item()
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

                    model_probs = model_probs[:trunc_index]
                    model_lower = model_lower[:trunc_index]
                    model_upper = model_upper[:trunc_index]

                    model_mrl = torch.trapz(model_probs, times[:trunc_index]).item()
                    abs_error = abs(true_rul_unit - model_mrl)
                    model_coverage = ((true_survival_probs[:trunc_index] >= model_lower) &
                                      (true_survival_probs[:trunc_index] <= model_upper)).float().mean().item()
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

# Aggregation
ae_df = pd.DataFrame(ae_details)
coverage_df = pd.DataFrame(coverage_details)
interval_length_df = pd.DataFrame(interval_length_details)
coverage_interval_ratio_df = pd.DataFrame(coverage_interval_ratio_details)

ranges = {
    "fm1(51,61)": range(51, 61),
    "fm2(111,121)": range(111, 121),
    "all": unit_list
}

def compute_summary(df, column):
    return df.groupby("Model")[column].mean().to_dict()

summary_tables = {
    "MAE": {},
    "Coverage": {},
    "Interval_Length": {},
    "Coverage_Interval_Ratio": {}
}

for key, unit_range in ranges.items():
    summary_tables["MAE"][key] = compute_summary(ae_df[ae_df["Unit"].isin(unit_range)], "AE")
    summary_tables["Coverage"][key] = compute_summary(coverage_df[coverage_df["Unit"].isin(unit_range)], "Coverage Ratio")
    summary_tables["Interval_Length"][key] = compute_summary(interval_length_df[interval_length_df["Unit"].isin(unit_range)], "Mean Interval Length")
    summary_tables["Coverage_Interval_Ratio"][key] = compute_summary(coverage_interval_ratio_df[coverage_interval_ratio_df["Unit"].isin(unit_range)], "Coverage/Mean Interval Length")

# Convert and save
pd.DataFrame(summary_tables["MAE"]).T.to_csv(f"MAE_2_{t}.csv", sep="\t")
# pd.DataFrame(summary_tables["Coverage"]).T.to_csv(f"Coverage_{t}.csv", sep="\t")
# pd.DataFrame(summary_tables["Interval_Length"]).T.to_csv(f"Interval_Length_{t}.csv", sep="\t")
# pd.DataFrame(summary_tables["Coverage_Interval_Ratio"]).T.to_csv(f"Coverage_Interval_Ratio_{t}.csv", sep="\t")
# Pivot tables (per-unit view)
ae_pivot_df = ae_df.pivot(index="Unit", columns="Model", values="AE")
# coverage_pivot_df = coverage_df.pivot(index="Unit", columns="Model", values="Coverage Ratio")
# interval_length_pivot_df = interval_length_df.pivot(index="Unit", columns="Model", values="Mean Interval Length")
# coverage_interval_ratio_pivot_df = coverage_interval_ratio_df.pivot(index="Unit", columns="Model", values="Coverage/Mean Interval Length")

# Save pivot tables
ae_pivot_df.to_csv(f"AE_Pivot_2_{t}.tsv", sep="\t")
# coverage_pivot_df.to_csv(f"Coverage_Pivot_{t}.tsv", sep="\t")
# interval_length_pivot_df.to_csv(f"Interval_Length_Pivot_{t}.tsv", sep="\t")
# coverage_interval_ratio_pivot_df.to_csv(f"Coverage_Interval_Ratio_Pivot_{t}.tsv", sep="\t")
