import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# === STEP 1: Settings ===
time_points = [20, 50, 75]
base_path = "./"  # Change this if files are in another directory
file_names = [f"{base_path}classes{t}.csv" for t in time_points]

# Map raw model column names to cleaned plot labels
model_map = {
    'CMGP_Cox': '$\mathbf{MFGPCox}$',
    'DeepBranched': 'DeepBrch',
    'NN_joint_ideal': 'NNJoint',
    'DeepSurv': 'DeepSurv'
}
excluded_model = 'NN_joint_mis'

# === STEP 2: Compute Errors and Save as CSV ===
error_file_paths = []
for t, file_path in zip(time_points, file_names):
    df = pd.read_csv(file_path)
    error_data = {}

    for model in model_map:
        # Get true column name (e.g., 'CMGP_Cox_p1' or 'CMGP_Cox_p2')
        # Convert true mode to integer index (0 for p1, 1 for p2)
        true_mode_index = df['Actual_Failure_Mode'].astype(int) - 1

        # Get model prediction matrix (N, 2) → columns: [p1, p2]
        probs = df[[f"{model}_p1", f"{model}_p2"]].to_numpy()

        # Use numpy indexing to get the correct predicted probability
        correct_probs = probs[np.arange(len(df)), true_mode_index]

        # Error = 1 - correct probability
        error_data[model_map[model]] = 1 - correct_probs
    # Save computed error file for later plotting
    error_df = pd.DataFrame(error_data)
    error_csv_path = f"{base_path}Absolute_Errors{t}_classes.csv"
    error_df.to_csv(error_csv_path, index=False)
    error_file_paths.append(error_csv_path)

# === STEP 3: Get global y-axis limits ===
global_y_min, global_y_max = float('inf'), float('-inf')
for path in error_file_paths:
    df = pd.read_csv(path)
    global_y_min = min(global_y_min, df.min().min())
    global_y_max = max(global_y_max, df.max().max())

# === STEP 4: Plotting Function ===
def plot_separate_boxplots(file_paths, time_points, model_names, colors, y_margin=0.1, global_y_min=None, global_y_max=None):
    if global_y_min is None or global_y_max is None:
        raise ValueError("global_y_min and global_y_max must be provided.")

    y_range = global_y_max - global_y_min
    y_min = global_y_min - y_margin * y_range
    y_max = global_y_max + y_margin * y_range

    fig, axes = plt.subplots(1, len(time_points), figsize=(6 * len(time_points), 10), sharey=True)
    if len(time_points) == 1:
        axes = [axes]

    for ax, file_path, t in zip(axes, file_paths, time_points):
        df = pd.read_csv(file_path)

        box = ax.boxplot(
            [df[model] for model in model_names],
            patch_artist=True,
            showmeans=True,
            meanprops=dict(marker='s', markerfacecolor='orange', markeredgecolor='black', markersize=8),
            boxprops=dict(color='black', linewidth=2),
            whiskerprops=dict(color='black', linewidth=1),
            capprops=dict(color='black', linewidth=1),
            medianprops=dict(color='black', linewidth=2),
            flierprops=dict(marker='o', markerfacecolor='red', markeredgecolor='black', markersize=8, linestyle='none')
        )

        for j, patch in enumerate(box['boxes']):
            model = model_names[j]
            if model == 'MFGPCox':
                patch.set_facecolor('lightgray')
            else:
                patch.set_facecolor('white')

        ax.set_xticklabels(model_names, rotation=90, fontsize=44)
        ax.set_title(f"$\\mathbf{{t^* = {t}}}$", fontsize=44, fontweight="bold")
        ax.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.6)
        ax.tick_params(axis='y', labelsize=44)

    axes[0].set_ylabel('Error', fontsize=44, labelpad=20, fontweight='bold')



    # fig.suptitle("Failure Mode Prediction", fontsize=44, fontweight='bold', y=1)

    plt.ylim(y_min, y_max)
    plt.tight_layout(rect=[0, 0, 1, 1])
    fig.text(0.5, 1.03, "Failure Mode", fontsize=44, fontweight='bold', ha='center')
    plt.savefig("AE_FM_boxplot_all.png", dpi=300, bbox_inches='tight')
    plt.show()


# === STEP 5: Call the plot function ===
plot_separate_boxplots(
    file_paths=error_file_paths,
    time_points=time_points,
    model_names=list(model_map.values()),
    colors=['white'] * len(model_map),
    global_y_min=global_y_min,
    global_y_max=global_y_max
)
