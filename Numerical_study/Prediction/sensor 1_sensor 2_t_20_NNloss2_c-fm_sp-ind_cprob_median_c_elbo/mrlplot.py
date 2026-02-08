import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_mrl_coverage_comparison(time_points, file_template, output_file, unit_groups):
    colors = ["lightgray", "white", "white"]
    bar_width = 0.03

    label_mapping = {
        "CMGP-Cox": "MFGPCox",
        "NN_joint(ideal)": "NNJoint",
        "NN_joint(mis)": "NNJ(Quad)",
    }

    excluded_models = {"NN_joint(mis)"}

    for group in unit_groups:
        fig, axes = plt.subplots(1, len(time_points), figsize=(7, 4.7), sharey=True)
        group_suffix = group['suffix']
        unit_range = group['range']
        all_models = set()
        data_by_time = {}

        for t_idx, t in enumerate(time_points):
            file_path = file_template.format(t)
            coverage_data = pd.read_csv(file_path, sep="\t")  # tab-separated!

            if "Unit" in coverage_data.columns and unit_range:
                coverage_data = coverage_data[coverage_data["Unit"].between(unit_range[0], unit_range[1])]

            mean_coverage = (
                coverage_data.groupby("Model")["MRL_Covered"].mean()
                if "MRL_Covered" in coverage_data.columns
                else coverage_data.set_index("Model")["MRL_Coverage_Proportion"]
            )

            filtered_models = mean_coverage[~mean_coverage.index.isin(excluded_models)]
            models = filtered_models.index.tolist()
            all_models.update(models)
            data_by_time[t] = filtered_models

        all_models = sorted(all_models)
        num_models = len(all_models)

        for t_idx, t in enumerate(time_points):
            ax = axes[t_idx]
            x_positions = np.arange(num_models) * (bar_width + 0.02)

            for m_idx, model in enumerate(all_models):
                value = data_by_time.get(t, pd.Series(index=all_models)).get(model, 0.0)
                ax.bar(
                    x_positions[m_idx], value, width=bar_width,
                    color=colors[m_idx % len(colors)], edgecolor="black", linewidth=2
                )

            ax.set_xticks(x_positions)
            ax.set_xticklabels(
                [label_mapping.get(model, model) for model in all_models],
                rotation=90, ha="center", fontsize=20
            )
            ax.set_title(f"$\\mathbf{{t^* = {t}}}$", fontsize=20, fontweight="bold", pad=10)
            ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.6)
            ax.set_ylim(0, 1.1)
            ax.axhline(y=1.0, color='black', linestyle='-', linewidth=0.5)
            ax.axhline(y=0.9, color='red', linestyle='--', linewidth=1.5)
            ax.tick_params(axis='y', labelsize=20)
            ax.set_yticks([0.0, 0.5, 0.9, 1])

        axes[0].set_ylabel("MRL Coverage Proportion", fontsize=20, labelpad=20, fontweight='bold')
        plt.tight_layout()
        group_output_file = output_file.format(group_suffix.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_'))
        plt.savefig(group_output_file, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()

# === Parameters ===
time_points = [20, 50, 75]
file_template = "./MRL_Coverage_Proportion_{}.tsv"
output_file = "./MRL_Coverage_Comparison_{}.png"
unit_groups = [
    {"range": None, "suffix": "All Units"},
]

# === Generate plots ===
plot_mrl_coverage_comparison(time_points, file_template, output_file, unit_groups)
