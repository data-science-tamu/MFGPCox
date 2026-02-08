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



t = 20
mrl_results = []

for unit_name in chain(range(51, 61), range(111, 121)):
    print(f"Processing unit: {unit_name}")

    tsv_file_path = rf"C:\Users\sina.aghaee\Desktop\GitLab\CM\numerical_study\numerical_study_19\sensor 1_sensor 2_t_{t}_NNloss2_c-fm_sp-ind_cprob_median_c_elbo\sp_{t}_exact_ns10000\{unit_name}.tsv"

    try:
        df = pd.read_csv(tsv_file_path, sep="\t")

        times = torch.tensor(df["Time"].values, dtype=torch.float32).to(device)
        true_survival_probs = torch.tensor(df["True_Survival_Probability"].values, dtype=torch.float32).to(device)
        cmgp_probs = torch.tensor(df["Survival_Probability"].values, dtype=torch.float32).to(device)
        cmgp_lower = torch.tensor(df["lower_bound"].values, dtype=torch.float32).to(device)
        cmgp_upper = torch.tensor(df["Upper_bound"].values, dtype=torch.float32).to(device)

        # Determine valid truncation index
        mask = (true_survival_probs > 0) & (cmgp_lower > 0)
        trunc_index = torch.where(mask == False)[0].min().item() if not mask.all() else len(times)
        trunc_index=200
        # Truncate all series
        times = times[:trunc_index]
        true_survival_probs = true_survival_probs[:trunc_index]
        cmgp_probs = cmgp_probs[:trunc_index]
        cmgp_lower = cmgp_lower[:trunc_index]
        cmgp_upper = cmgp_upper[:trunc_index]

        true_mrl = true_rul_dict[unit_name]
        pred_mrl = torch.trapz(cmgp_probs, times).item()
        lower_mrl = torch.trapz(cmgp_lower, times).item()
        upper_mrl = torch.trapz(cmgp_upper, times).item()

        mrl_results.append({
            "Unit": unit_name,
            "Model": "CMGP-Cox",
            "True_MRL": true_mrl,
            "Pred_MRL": pred_mrl,
            "Lower_MRL": lower_mrl,
            "Upper_MRL": upper_mrl
        })

        # Additional models like NNJoint, etc.
        for col in df.columns:
            if col.startswith("est_sz_mean_"):
                model_name = col.replace("est_sz_mean_", "")
                model_probs = torch.tensor(df[col].values, dtype=torch.float32).to(device)
                lower_col = col.replace("mean", "lower")
                upper_col = col.replace("mean", "upper")

                if lower_col in df.columns and upper_col in df.columns:
                    model_lower = torch.tensor(df[lower_col].values, dtype=torch.float32).to(device)
                    model_upper = torch.tensor(df[upper_col].values, dtype=torch.float32).to(device)

                    model_probs = model_probs[:trunc_index]
                    model_lower = model_lower[:trunc_index]
                    model_upper = model_upper[:trunc_index]

                    model_mrl = torch.trapz(model_probs, times).item()
                    model_lower_mrl = torch.trapz(model_lower, times).item()
                    model_upper_mrl = torch.trapz(model_upper, times).item()

                    mrl_results.append({
                        "Unit": unit_name,
                        "Model": model_name,
                        "True_MRL": true_mrl,
                        "Pred_MRL": model_mrl,
                        "Lower_MRL": model_lower_mrl,
                        "Upper_MRL": model_upper_mrl
                    })

    except FileNotFoundError:
        print(f"File not found for unit: {unit_name}")
    except Exception as e:
        print(f"Error processing unit {unit_name}: {e}")

# Save the output
mrl_results_df = pd.DataFrame(mrl_results)
output_path = f"MRL_Results_with_Confidence_{t}.tsv"
mrl_results_df.to_csv(output_path, sep="\t", index=False)
print(f"\nSaved: {output_path}")

# Compute coverage indicator for each row
mrl_results_df["MRL_Covered"] = (
    (mrl_results_df["True_MRL"] >= mrl_results_df["Lower_MRL"]) &
    (mrl_results_df["True_MRL"] <= mrl_results_df["Upper_MRL"])
).astype(int)

# Group by model and compute the coverage proportion
mrl_coverage_df = (
    mrl_results_df
    .groupby("Model")["MRL_Covered"]
    .mean()
    .reset_index()
    .rename(columns={"MRL_Covered": "MRL_Coverage_Proportion"})
)

# Save to file
coverage_output_path = f"MRL_Coverage_Proportion_2_{t}.tsv"
mrl_coverage_df.to_csv(coverage_output_path, sep="\t", index=False)
print(f"Saved: {coverage_output_path}")



# Ensure numerical precision handling
epsilon = 1e-5

# Filter CMGP-Cox entries
cmgp_df = mrl_results_df[mrl_results_df["Model"] == "CMGP-Cox"].copy()

# Identify units where true MRL is outside the confidence interval
not_covered_mask = ~(
    (cmgp_df["True_MRL"] >= cmgp_df["Lower_MRL"] - epsilon) &
    (cmgp_df["True_MRL"] <= cmgp_df["Upper_MRL"] + epsilon)
)

not_covered_units = cmgp_df[not_covered_mask]["Unit"].tolist()

print(f"Units not covered by CMGP-Cox interval: {not_covered_units}")
