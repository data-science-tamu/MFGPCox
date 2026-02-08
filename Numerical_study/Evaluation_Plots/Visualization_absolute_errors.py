import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def plot_separate_boxplots(file_paths, time_points, model_names, colors, y_margin=0.1, global_y_min=None,
                           global_y_max=None):
    if global_y_min is None or global_y_max is None:
        raise ValueError("global_y_min and global_y_max must be provided to ensure consistent y-axis limits.")

    y_range = global_y_max - global_y_min
    y_min = global_y_min - y_margin * y_range
    y_max = global_y_max + y_margin * y_range

    fig, axes = plt.subplots(1, len(time_points), figsize=(6 * len(time_points), 10), sharey=True)

    if len(time_points) == 1:
        axes = [axes]  # Ensure axes is iterable when there's only one subplot

    excluded_models = {"NN_joint(mis)"}  # Models to exclude

    for ax, file_path, t in zip(axes, file_paths, time_points):
        data = pd.read_csv(file_path)

        # Remove excluded models
        filtered_model_names = [model for model in model_names if model not in excluded_models]

        box = ax.boxplot(
            [data[model] for model in filtered_model_names],
            patch_artist=True,
            showmeans=True,
            meanprops=dict(marker='s', markerfacecolor='orange', markeredgecolor='black', markersize=8),
            boxprops=dict(color='black', linewidth=2),
            whiskerprops=dict(color='black', linewidth=1),
            capprops=dict(color='black', linewidth=1),
            medianprops=dict(color='black', linewidth=2),
            flierprops=dict(marker='o', markerfacecolor='red', markeredgecolor='black', markersize=8, linestyle='none')
        )

        for j, (patch, color) in enumerate(zip(box['boxes'], colors)):
            if filtered_model_names[j] == 'CMGP-Cox':
                patch.set_facecolor('lightgray')
            elif filtered_model_names[j] in ['DeepBranched', 'Cox']:
                patch.set_facecolor('white')
            else:
                patch.set_facecolor(color)

        ax.set_xticklabels(
            ['$\mathbf{MFGPCox}$', 'DeepBrch', 'NNJoint', 'DeepSurv', 'CoxPH'],  # Updated labels without NNJ(Quad)
            rotation=90,
            fontsize=44,
            # fontweight='bold'
        )

        ax.set_title(f"$\\mathbf{{t^* = {t}}}$", fontsize=44 ,fontweight="bold")
        ax.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.6)
        ax.tick_params(axis='y', labelsize=44)

    axes[0].set_ylabel('Absolute Error', fontsize=44, labelpad=20, fontweight='bold')

    fig.subplots_adjust(top=0.8)  # Adjust the top margin
    plt.tight_layout(rect=[0, 0, 1, 1])  # Reserve space for the legend
    # fig.legend([
    #     plt.Line2D([], [], color='black', linewidth=4),
    #     plt.Line2D([], [], marker='s', color='orange', markeredgecolor='black', markersize=16, linestyle='None'),
    #     plt.Line2D([], [], marker='o', color='red', markeredgecolor='black', markersize=16, linestyle='None')
    # ], ['Median', 'Mean', 'Outlier'], loc='upper center', frameon=False, ncol=3, bbox_to_anchor=(0.5, 1.05),
    #     prop={'size': 44, 'weight': 'bold'})

    plt.ylim(y_min, y_max)
    plt.savefig("AE_boxplot_all.png", dpi=300)
    plt.show()


# Define file paths and global y-limits
time_points = [20, 50, 75]
file_paths = [f"./Absolute_Errors{t}.csv" for t in time_points]
model_names = ['CMGP-Cox', 'DeepBranched', 'NN_joint(ideal)', 'NN_joint(mis)', 'DeepSurv', 'Cox']
colors = ['white', 'white', 'white', 'white', 'white', 'white']

# Remove excluded model from global y-range calculation
filtered_model_names = [model for model in model_names if model not in {"NN_joint(mis)"}]

global_y_min, global_y_max = float('inf'), float('-inf')
for file_path in file_paths:
    data = pd.read_csv(file_path)
    global_y_min = min(global_y_min, data[filtered_model_names].min().min())
    global_y_max = max(global_y_max, data[filtered_model_names].max().max())

plot_separate_boxplots(file_paths, time_points, filtered_model_names, colors, y_margin=0.1,
                       global_y_min=global_y_min, global_y_max=global_y_max)
