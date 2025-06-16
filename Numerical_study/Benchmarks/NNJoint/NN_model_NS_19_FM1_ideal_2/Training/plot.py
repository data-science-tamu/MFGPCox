import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV files
noise_data = pd.read_csv("training_noise.csv")
no_noise_data = pd.read_csv("training_no_noise.csv")

# Filter for ID = 1
noise_id1 = noise_data[noise_data['id'] == 50]
no_noise_id1 = no_noise_data[no_noise_data['id'] == 50]

# Degradation column indices (2 to 9)
degradation_indices = list(range(2, 4))

# Plotting
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 10))
axes = axes.flatten()

for i, idx in enumerate(degradation_indices):
    col_name = noise_id1.columns[idx]
    axes[i].plot(noise_id1.iloc[:, 1], noise_id1.iloc[:, idx], label='With Noise', alpha=0.7)
    axes[i].plot(no_noise_id1.iloc[:, 1], no_noise_id1.iloc[:, idx], label='No Noise', linestyle='--')
    axes[i].set_title(f'Column {idx} - ID 1')
    axes[i].legend()
    axes[i].set_xlabel('Time')
    axes[i].set_ylabel('Value')

plt.tight_layout()
plt.show()
