import matplotlib.pyplot as plt
import pandas as pd
from itertools import chain
from matplotlib.patches import Patch

cutoff = 126
times = [20, 50, 75]

def align_data_with_time(data, time_column, value_columns, x_range=150):
    full_time = pd.DataFrame({time_column: range(1, x_range + 1)})
    aligned_data = pd.merge(full_time, data, on=time_column, how='left')
    return aligned_data

def plot_survival_probabilities_grid(units, output_dir, times):
    x_range = 150
    models = ["MFGP-C", "NN-J-E"]  # Removed "NN-J-Q"
    colors = ["#A9A9A9", "#FF8A7B"]
    line_styles = ['-', '-']

    model_name_mapping = {
        "MFGP-C": "$\mathbf{MFGPCox}$",
        "NN-J-E": "NNJoint"
    }


    fig, axes = plt.subplots(2, 3, figsize=(22, 11), sharex=True, sharey=True)  # Adjusted to (2,3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, model in enumerate(models):
        display_name = model_name_mapping[model]
        for j, t in enumerate(times):
            ax = axes[i, j]
            for unit_name in units:
                tsv_file_path = rf"../Prediction/sensor 1_sensor 2_t_{t}_NNloss2_c-fm_sp-ind_cprob_median_c_elbo/sp_{t}_exact_ns10000/{unit_name}.tsv"
                try:
                    df = pd.read_csv(tsv_file_path, sep="\t")
                    df_aligned = align_data_with_time(df, "Time", df.columns, x_range)
                    time = df_aligned["Time"].values
                    true_survival_probs = df_aligned["True_Survival_Probability"].values

                    if model == "MFGP-C":
                        model_probs = df_aligned["Survival_Probability"].values
                        model_lower = df_aligned["lower_bound"].values
                        model_upper = df_aligned["Upper_bound"].values
                    elif model == "NN-J-E" and "est_sz_mean_NN_joint(ideal)" in df.columns:
                        model_probs = df_aligned["est_sz_mean_NN_joint(ideal)"].values
                        model_lower = df_aligned["est_sz_lower_NN_joint(ideal)"].values
                        model_upper = df_aligned["est_sz_upper_NN_joint(ideal)"].values
                    else:
                        continue

                    ax.plot(time, true_survival_probs, label="True", color="#2CA02C", linewidth=4, linestyle="-.")
                    ax.plot(time, model_probs, label=model, color=colors[i], linestyle=line_styles[i], linewidth=4, alpha=0.8)
                    ax.fill_between(time, model_lower, model_upper, color=colors[i], alpha=0.2)
                    ax.axvline(x=t, color="red", linestyle="--", linewidth=4)
                except FileNotFoundError:
                    print(f"File not found for unit: {unit_name}")
                except Exception as e:
                    print(f"Error processing unit {unit_name}: {e}")

            if j == 0:
                ax.set_ylabel(display_name, fontsize=50, rotation=90, labelpad=50, va='center')
            if i == 0:
                ax.set_title(
                    f"$\\mathbf{{t^* = {t}}}$",
                    ha='center',
                    va='bottom',
                    fontsize=50,
                    fontweight='bold'
                )

            ax.tick_params(axis='both', which='major', labelsize=45)
            ax.set_xlim(1, x_range)
            ax.set_xticks([20, 50, 75, 105, 140])
            ax.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.6)

    # Add a legend outside the subplots
    handles = [
        plt.Line2D([0], [0], color="#2CA02C", linestyle="-.", linewidth=4, label="True Survival Probability"),
        plt.Line2D([0], [0], color="black", linestyle="-", linewidth=4, label="Predicted Survival Probability"),
        Patch(color="black", alpha=0.22, label="Credible Interval")
    ]

    fig.legend(
        handles=handles[:2],  # First two items
        loc="upper center",
        fontsize=45,
        frameon=False,
        ncol=2,  # Two items in one row
        bbox_to_anchor=(0.5, 1.18)
    )

    fig.legend(
        handles=[handles[2]],  # Third item
        loc="upper center",
        fontsize=45,
        frameon=False,
        ncol=1,  # Single item in one row
        bbox_to_anchor=(0.17, 1.11)
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.89, bottom=0.15)

    plt.savefig(f"{output_dir}/Survival_Probabilities_Grid_{unit[0]}.png", bbox_inches='tight', dpi=300)
    plt.show()

# units = chain(range(51, 61), range(111, 121))
units = [[55], [59]]
output_dir = r"./"
for unit in units:
    plot_survival_probabilities_grid(unit, output_dir, times)
