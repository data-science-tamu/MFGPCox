import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Load the data
file_path_fm1 = './all_units_time_series_fm1.csv'
file_path_fm2 = './all_units_time_series_fm2.csv'
data_fm1 = pd.read_csv(file_path_fm1)
data_fm2 = pd.read_csv(file_path_fm2)

# Helper: Filter units based on time_limit
def filter_units(data, time_limit):
    if time_limit is None:
        valid_units = list(range(1, 51)) + list(range(61, 111))
    else:
        valid_units = list(range(51, 61)) + list(range(111, 122))
    data_filtered = data[data['unit number'].isin(valid_units)]
    if time_limit is not None:
        data_filtered = data_filtered[data_filtered['time, in cycles'] <= time_limit]
    return data_filtered

# Main plotting function
def plot_sensor_data_grid(data_fm1, data_fm2, sensors, time_limits, colors, output_dir, title_suffix=""):
    time_limits = [None, 20, 50, 75]
    fm_styles = {
        "FM 1": {"color": "#1f77b4", "linestyle": "-"},
        "FM 2": {"color": "#ff9999", "linestyle": "-"}
    }

    fig, axes = plt.subplots(2, 4, figsize=(25, 12), sharex=False, sharey=False)
    fig.subplots_adjust(hspace=0.4, wspace=0.3)

    for row, sensor in enumerate(sensors):
        for col, time_limit in enumerate(time_limits):
            ax = axes[row, col]

            fm1_data = filter_units(data_fm1, time_limit)
            fm2_data = filter_units(data_fm2, time_limit)

            alpha1 = 0.4 if time_limit is None else 0.8
            alpha2 = 0.3 if time_limit is None else 0.8

            for unit in fm1_data['unit number'].unique():
                unit_data = fm1_data[fm1_data['unit number'] == unit]
                ax.plot(unit_data['time, in cycles'], unit_data[sensor],
                        color=fm_styles["FM 1"]["color"], linestyle=fm_styles["FM 1"]["linestyle"],
                        alpha=alpha1, linewidth=2.5,
                        label='FM 1' if unit == fm1_data['unit number'].unique()[0] else "")
                if time_limit is None:
                    ax.scatter(unit_data['time, in cycles'].iloc[-1], unit_data[sensor].iloc[-1],
                               color='black', marker='X', s=100)

            for unit in fm2_data['unit number'].unique():
                unit_data = fm2_data[fm2_data['unit number'] == unit]
                ax.plot(unit_data['time, in cycles'], unit_data[sensor],
                        color=fm_styles["FM 2"]["color"], linestyle=fm_styles["FM 2"]["linestyle"],
                        alpha=alpha2, linewidth=2.5,
                        label='FM 2' if unit == fm2_data['unit number'].unique()[0] else "")
                if time_limit is None:
                    ax.scatter(unit_data['time, in cycles'].iloc[-1], unit_data[sensor].iloc[-1],
                               color='black', marker='X', s=100)

            if time_limit is not None:
                ax.axvline(x=time_limit, color='red', linestyle='--', linewidth=5, alpha=0.8)

            ax.grid(axis='y', linestyle='--', linewidth=1.2, alpha=0.6)
            ax.tick_params(axis='both', which='major', labelsize=32, width=2, length=6)

            ticks_map = {
                (0, 1): [0, 5, 10, 15, 20],
                (0, 2): [0, 10, 20, 30, 40, 50],
                (0, 3): [0, 25, 50, 75],
                (1, 1): [0, 5, 10, 15, 20],
                (1, 2): [0, 10, 20, 30, 40, 50],
                (1, 3): [0, 25, 50, 75]
            }
            ax.set_xticks(ticks_map.get((row, col), [0, 20, 50, 75, 125]))

            if row == 0:
                ax.set_title("Until Event Time" if time_limit is None else f"$\\mathbf{{t^* = {time_limit}}}$",
                             fontsize=33, fontweight='bold', pad=15)

            if col == 0:
                ax.set_ylabel(sensor.capitalize(), fontsize=38, fontweight='bold', labelpad=15)

    # Global legend
    # Global legend
    handles = [
        plt.Line2D([0], [0], color=fm_styles["FM 1"]["color"], linestyle=fm_styles["FM 1"]["linestyle"], linewidth=6,
                   label="Failure Mode 1 (k=1)"),
        plt.Line2D([0], [0], color=fm_styles["FM 2"]["color"], linestyle=fm_styles["FM 2"]["linestyle"], linewidth=6,
                   label="Failure Mode 2 (k=2)"),
        plt.Line2D([0], [0], color='black', marker='X', linestyle='', markersize=30, label="Event Time")
    ]
    fig.legend(handles=handles, loc="upper center", fontsize=38, frameon=False, ncol=4, bbox_to_anchor=(0.5, 1.035))

    # Add "Time" and subplot labels to each column
    label_positions = [0.19, 0.42, 0.66, 0.9]  # for (a), (b), (c), (d)
    time_label_positions = [0.258, 0.495, 0.73, 0.968]  # slightly different for "Time"

    for i, (label_x, time_x) in enumerate(zip(label_positions, time_label_positions)):
        fig.text(time_x, 0.04, 'Time', ha='center', fontsize=36)
        fig.text(label_x, -0.02, f"({chr(97 + i)})", ha='center', fontsize=40, fontweight='bold')

    # Save
    plt.tight_layout(rect=[0.02, 0.07, 1, 0.93])
    plt.savefig(f"{output_dir}/Generated_Sensor_Data{title_suffix}.png", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


# Run
sensors = ['sensor 1', 'sensor 2']
colors = {1: 'blue', 2: 'red'}
output_directory = "./"

plot_sensor_data_grid(data_fm1, data_fm2, sensors, [20, 50, 75], colors, output_directory, title_suffix="")
