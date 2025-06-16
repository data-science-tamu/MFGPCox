import torch
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = "cpu"

plot_dir = 'fm2_data/time_series_plots'
os.makedirs(plot_dir, exist_ok=True)

summary_plot_dir = 'fm2_data/summary_plots'
os.makedirs(summary_plot_dir, exist_ok=True)

# Define constants and distributions
sigma_j1 = 0.8
sigma_j2 = 0.7

mu_1 = torch.tensor([2.3, 0.12, 0.015], device=device)
mu_2 = torch.tensor([1.6, 0.1, 0.01], device=device)

scaling_factor = 0.6

Sigma_1 = torch.tensor([[0.7 * scaling_factor, 3e-5, 4e-5],
                        [3e-5, 2e-5 * scaling_factor, 1e-7],
                        [4e-5, 1e-7, 3e-6 * scaling_factor]], device=device)


Sigma_2 = torch.tensor([[0.1 * scaling_factor, 3e-5, 4e-5],
                        [3e-5, 2e-5 * scaling_factor, 1e-7],
                        [4e-5, 1e-7, 3e-6 * scaling_factor]], device=device)

# Small epsilon to avoid numerical issues in covariance matrices
epsilon = 15e-7
Sigma_1 += torch.eye(Sigma_1.shape[0], device=device) * epsilon
Sigma_2 += torch.eye(Sigma_2.shape[0], device=device) * epsilon



# lists to store samples
B_i1_samples_list = []
B_i2_samples_list = []
failure_times_list = []
Y1_series = []
Y2_series = []

# Define the Weibull baseline hazard function parameters
# alpha = 1.05
# lambda_ = 0.0001

b = -10
rho = 0.0095

beta_1 = 0.018
beta_2 = 0.01


def Z1(t):
    return torch.stack([
        torch.ones_like(t),
        t ** 0.8 * torch.cos(t),
        t ** 2 + t
    ], dim=-1)  # Shape: [T, 3]


def Z2(t):
    return torch.stack([torch.ones_like(t), t ** 1.5, t ** 2], dim=-1)  # Shape: [T, 3]



# def get_baseline_hazard(t):
#     return lambda_ * alpha * (t ** (alpha - 1))  # Shape: [T]

def get_baseline_hazard(t):
    return torch.exp(b + rho * t)  # Shape: [T]  # Shape: [T]


time = torch.linspace(0, 200, steps=1000)  # Shape: [T], 1000 points between 0 and 200

# Avoid division by zero by ensuring no zero values in time
time[time == 0] = 1e-8

# Compute the baseline hazard values
baseline_hazard = get_baseline_hazard(time)

# Plot the baseline hazard over time
plt.figure(figsize=(8, 6))
plt.plot(time.numpy(), baseline_hazard.numpy(), label='Baseline Hazard')
plt.xlabel('Time')
plt.ylabel('Hazard')
plt.title('Baseline Hazard Function Over Time')
plt.grid(True)
plt.legend()
plot_filename = os.path.join(summary_plot_dir, 'Baseline_Hazard.png')
plt.savefig(plot_filename)
plt.show()


def true_hazard(t, B_i1_samples, B_i2_samples):
    Z1_t = Z1(t)  # Shape: [T, 3]
    Z2_t = Z2(t)  # Shape: [T, 3]

    Z1_t = Z1_t.unsqueeze(0)  # Shape: [1, T, 3]
    Z2_t = Z2_t.unsqueeze(0)  # Shape: [1, T, 3]

    Z1_t = Z1_t.expand(B_i1_samples.shape[0], -1, -1)  # Shape: [N, T, 3]
    Z2_t = Z2_t.expand(B_i2_samples.shape[0], -1, -1)  # Shape: [N, T, 3]

    ZT1_Bi1 = torch.matmul(Z1_t, B_i1_samples.unsqueeze(-1)).squeeze(-1)  # Shape: [N, T]
    ZT2_Bi2 = torch.matmul(Z2_t, B_i2_samples.unsqueeze(-1)).squeeze(-1)  # Shape: [N, T]

    baseline_hazard = get_baseline_hazard(t)  # Shape: [T]
    baseline_hazard = baseline_hazard.unsqueeze(0).expand(B_i1_samples.shape[0], -1)  # Shape: [N, T]

    # hazard = baseline_hazard * torch.exp((0.06 * ZT1_Bi1) + (0.05 * ZT2_Bi2))  # Shape: [N, T]
    hazard = baseline_hazard * torch.exp((beta_1 * ZT1_Bi1) + (beta_2 * ZT2_Bi2))  # Shape: [N, T]

    return hazard


def true_survival(t, B_i1, B_i2, t_star, num_points=1000):
    ls = torch.linspace(t_star, t.squeeze(0), num_points).to(device)
    vals = true_hazard(ls, B_i1, B_i2)
    integral_approx = torch.trapz(vals, ls)
    survival = torch.exp(-integral_approx)
    return survival  # Shape: [N]


def f(t, B_i1_samples, B_i2_samples, t_star, num_points=1000):
    hazard = true_hazard(t, B_i1_samples, B_i2_samples)  # Shape: [N, T]
    survival = true_survival(t, B_i1_samples, B_i2_samples, t_star, num_points)  # Shape: [N]
    f_t = hazard.squeeze(1) * survival  # Shape: [N, T]
    return f_t


def compute_f_for_points(B_i1_samples, B_i2_samples, t_star, num_points=1000):
    t_values = torch.linspace(0.1, 200, num_points, device=device)
    f_values = []
    for t in t_values:
        f_t = f(t, B_i1_samples, B_i2_samples, t_star, num_points)
        f_values.append(f_t)
    f_values = torch.stack(f_values, dim=0)
    return t_values, f_values


##############################
#
# B_i1_samples = torch.distributions.MultivariateNormal(mu_1, Sigma_1).sample((N,))
# B_i2_samples = torch.distributions.MultivariateNormal(mu_2, Sigma_2).sample((N,))
# # Example of using the function and plotting
# t_values, f_values = compute_f_for_points(B_i1_samples, B_i2_samples, 0)
#
# # Choose the first signal (index 0) from N different signals
# signal_index = 0
# f_signal = f_values[:, signal_index]  # Extract f(t) for the chosen signal
#
# # Plotting
# plt.figure(figsize=(8, 6))
# plt.plot(t_values.cpu().numpy(), f_signal.cpu().numpy(), label=f'Signal {signal_index}')
# plt.xlabel('Time')
# plt.ylabel('f(t)')
# plt.title(f'Density Function for Signal {signal_index}')
# plt.legend()
# plt.grid(True)
# plt.show()

# print()


B_i1_samples = torch.distributions.MultivariateNormal(mu_1, Sigma_1).sample()
B_i2_samples = torch.distributions.MultivariateNormal(mu_1, Sigma_1).sample()

B_i1_samples = B_i1_samples.unsqueeze(0)
B_i2_samples = B_i2_samples.unsqueeze(0)

N = B_i1_samples.shape[0]
t_min = torch.tensor(0., device=device)
t_max = torch.tensor(200., device=device)
t_star = t_min

# Compute t_values and f_values
t_values = torch.linspace(t_min, t_max, 1000, device=device)
with torch.no_grad():
    f_values = torch.zeros(1000, device=device)
    for idx, t in enumerate(t_values):
        t = t.unsqueeze(-1)
        f_values[idx] = torch.max(f(t, B_i1_samples, B_i2_samples, t_star, 1000))
M = torch.max(f_values) * 1.1

# Rejection Sampling
failure_times = torch.zeros(10000, dtype=torch.float32, device=device)  # Collecting 1000 samples
num_samples = 0
max_attempts = 1000  # Set a maximum number of attempts per unit

while num_samples < 10000:
    accepted = False
    attempts = 0
    while not accepted and attempts < max_attempts:
        t_prime = (t_max - t_min) * torch.rand(1, device=device) + t_min
        t_prime = torch.clamp(t_prime, min=0.1)

        f_t_prime = f(t_prime, B_i1_samples[0:1], B_i2_samples[0:1], t_star, 1000).to(device)
        u = torch.rand(1, device=device)

        if u <= f_t_prime / M:
            failure_times[num_samples] = t_prime
            accepted = True
        attempts += 1
    if not accepted:
        raise RuntimeError(f"Failed to generate failure time after {max_attempts} attempts.")
    num_samples += 1

t_values, f_values = compute_f_for_points(B_i1_samples, B_i2_samples, 0)

failure_times_cpu = failure_times.cpu().numpy()
t_values_cpu = t_values.cpu().numpy()
f_values_cpu = f_values[:, 0].cpu().numpy()  # Assuming we are plotting for signal index 0

plt.figure(figsize=(10, 6))

plt.plot(t_values_cpu, f_values_cpu, label='Density Function f(t)', color='blue')

plt.hist(failure_times_cpu, bins=30, density=True, alpha=0.6, color='orange', label='Histogram of Samples')

uniform_density = (M).cpu().numpy()
plt.axhline(y=uniform_density, color='green', linestyle='--', label='Uniform Density M*g(t)')

plt.xlabel('Time')
plt.ylabel('Density')
plt.title('Density Function, Histogram of Samples, and Uniform Density - sample unit from failure 2')
plt.legend()
plt.grid(True)
plt.show()



N = 60


##################################
def generate_failure_times(t_min, t_max, B_i1_samples, B_i2_samples, t_star, num_points=1000, device=device):
    N = B_i1_samples.shape[0]
    t_min = t_min.to(device)
    t_max = t_max.to(device)
    B_i1_samples = B_i1_samples.to(device)
    B_i2_samples = B_i2_samples.to(device)
    t_star = t_star.to(device)
    # M = torch.tensor(M, dtype=torch.float32).to(device)

    t_values = torch.linspace(t_min, t_max, num_points, device=device)
    f_values = torch.zeros(num_points, device=device)

    for idx, t in enumerate(t_values):
        t = t.unsqueeze(-1)
        f_values[idx] = torch.max(f(t, B_i1_samples, B_i2_samples, t_star, num_points))

    M = torch.max(f_values) * 1.1
    failure_times = torch.zeros(N, dtype=torch.float32, device=device)

    for i in range(N):
        accepted = False
        attempts = 0
        max_attempts = 1000  # Set a maximum number of attempts per unit
        while not accepted and attempts < max_attempts:
            t_prime = (t_max - t_min) * torch.rand(1, device=device) + t_min
            t_prime = torch.clamp(t_prime, min=0.1)

            f_t_prime = f(t_prime, B_i1_samples[i:i + 1], B_i2_samples[i:i + 1], t_star, num_points).to(device)
            u = torch.rand(1, device=device)

            if u <= f_t_prime / M:
                failure_times[i] = t_prime
                accepted = True
            attempts += 1
        if not accepted:
            raise RuntimeError(f"Failed to generate failure time for unit {i + 1} after {max_attempts} attempts.")
    return failure_times


def epsilon_j(t, sigma_j):
    return torch.normal(mean=0, std=sigma_j, size=t.shape, device=t.device)


def compute_Y_ij(t, B_i_samples, Z_func, sigma_j):
    Z_t = Z_func(t)  # Shape: [T, 3]
    B_i = B_i_samples.unsqueeze(-1)  # Shape: [3, 1]
    ZT_Bi = torch.matmul(Z_t, B_i).squeeze(-1)  # Shape: [T]
    epsilon = epsilon_j(t, sigma_j)  # Shape: [T]
    Y_ij = ZT_Bi + epsilon  # Shape: [T]
    return Y_ij  # Shape: [T]


t_min = torch.tensor(0.0, device=device)
t_max = torch.tensor(200.0, device=device)

density_folder = 'fm2_data/density_plots'
hazard_folder = 'fm2_data/hazard_plots'
survival_folder = 'fm2_data/survival_plots'

# Create directories if they don't exist
os.makedirs(density_folder, exist_ok=True)
os.makedirs(hazard_folder, exist_ok=True)
os.makedirs(survival_folder, exist_ok=True)

# Generate data for each unit
for i in range(N):
    max_attempts = 100 # To prevent infinite loops
    attempts = 0
    while attempts < max_attempts:
        attempts += 1
        # Generate B_i1_sample and B_i2_sample
        B_i1_sample = torch.distributions.MultivariateNormal(mu_1, Sigma_1).sample()
        B_i2_sample = torch.distributions.MultivariateNormal(mu_2, Sigma_2).sample()

        # Generate failure time for unit i
        failure_time = generate_failure_times(t_min, t_max, B_i1_sample.unsqueeze(0), B_i2_sample.unsqueeze(0),
                                              torch.tensor(0.0), num_points=1000, device=device)

        # Check if failure time is smaller than 1
        if failure_time[0] < 1:
            print(f"Unit {i}: Failure time {failure_time[0]} is smaller than 1, regenerating...")
            continue
        # Generate time series
        t_i = torch.cat((torch.arange(1, failure_time[0], device=device),
                         torch.tensor([failure_time[0]], device=device)))
        Y1_i = compute_Y_ij(t_i, B_i1_sample, Z1, sigma_j1)
        Y2_i = compute_Y_ij(t_i, B_i2_sample, Z2, sigma_j2)

        # Check if signals are increasing
        # if Y1_i[0] <= Y1_i[-1] and Y2_i[0] <= Y2_i[-1]:
        # if Y1_i[0] + 20 <= Y1_i[-1] and Y2_i[0] + 20 <= Y2_i[-1]:
        # Store the samples
        B_i1_samples_list.append(B_i1_sample)
        B_i2_samples_list.append(B_i2_sample)
        failure_times_list.append(failure_time[0])
        Y1_series.append(Y1_i)
        Y2_series.append(Y2_i)

        t_values, f_values = compute_f_for_points(B_i1_sample.unsqueeze(0), B_i2_sample.unsqueeze(0), 0)

        f_signal = f_values  # Extract f(t) for the chosen signal

        plt.figure(figsize=(8, 6))
        plt.plot(t_values.cpu().numpy(), f_signal.cpu().numpy(), label=f'Signal {i + 1}')
        plt.xlabel('Time')
        plt.ylabel('f(t)')
        plt.title(f'Density Function for Signal {i + 1}')
        plt.legend()
        plt.grid(True)
        density_path = os.path.join(density_folder, f'density_signal_{i + 1}.png')
        plt.savefig(density_path)
        plt.close()  # Close the figure after saving

        # Survival Probability Plot
        survival_probabilities = []
        for t in t_values:
            survival_prob = true_survival(t, B_i1_sample.unsqueeze(0), B_i2_sample.unsqueeze(0), 0, 1000)
            survival_probabilities.append(survival_prob)

        survival_probabilities = torch.tensor(survival_probabilities)

        plt.figure(figsize=(8, 6))
        plt.plot(t_values.cpu().numpy(), survival_probabilities.cpu().numpy(), label=f'Signal {i + 1}')
        plt.xlabel('Time')
        plt.ylabel('Survival Probability')
        plt.title('Survival Probability Over Time')
        plt.legend()
        plt.grid(True)
        survival_path = os.path.join(survival_folder, f'survival_signal_{i + 1}.png')
        plt.savefig(survival_path)
        plt.close()

        # Hazard Rate Plot
        true_hazard_values = []
        for t in t_values:
            hazard_rate = true_hazard(t, B_i1_sample.unsqueeze(0), B_i2_sample.unsqueeze(0))
            true_hazard_values.append(hazard_rate)

        true_hazard_rates = torch.tensor(true_hazard_values)

        plt.figure(figsize=(8, 6))
        plt.plot(t_values.cpu().numpy(), true_hazard_rates.cpu().numpy(), label=f'Signal {i + 1}')
        plt.xlabel('Time')
        plt.ylabel('Hazard Rate')
        plt.title('Hazard Rate Over Time')
        plt.ylim(0, 5)
        plt.legend()
        plt.grid(True)
        hazard_path = os.path.join(hazard_folder, f'hazard_signal_{i + 1}.png')
        plt.savefig(hazard_path)
        plt.close()

        break
    else:
        raise RuntimeError(f"Failed to generate increasing signals for unit {i + 1} after {max_attempts} attempts.")

# Convert lists to tensors
B_i1_samples = torch.stack(B_i1_samples_list)
B_i2_samples = torch.stack(B_i2_samples_list)
failure_times = torch.stack(failure_times_list)

# Plotting


# Loop through all units and save the plot for each one
for i in range(N):
    plt.figure(figsize=(10, 6))

    plt.plot(Y1_series[i].cpu().numpy(), label='Sensor 1 (j=1)')
    plt.plot(Y2_series[i].cpu().numpy(), label='Sensor 2 (j=2)')

    plt.xlabel('Time')
    plt.ylabel('Y_i,j(t)')
    plt.title(f'Time Series for Unit {i + 1}')
    plt.legend()
    plt.grid(True)

    plot_filename = os.path.join(plot_dir, f'unit_{i + 1}_time_series.png')
    plt.savefig(plot_filename)
    plt.close()
#
print(f"Plots saved to {plot_dir}")

t = np.linspace(0, 20, 1000)

# Z1 components
Z1_component1 = np.ones_like(t)
Z1_component2 = t ** 0.7 * np.sin(t)
Z1_component3 = t ** 2

# Z2 components
Z2_component1 = np.ones_like(t)
Z2_component2 = t ** 1.5
Z2_component3 = t ** 2

# Plot all units for Sensor 1 (j=1) on the same plot
plt.figure(figsize=(10, 6))
for i in range(N):
    plt.plot(Y1_series[i].cpu().numpy(), label=f'Unit {i + 1}')
plt.xlabel('Time')
plt.ylabel('Y_i,j(t)')
plt.title('Time Series for All Units (Sensor 1)')
plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1))
plt.grid(True)

# Save the plot
plot_filename = os.path.join(summary_plot_dir, 'all_units_sensor1.png')
plt.savefig(plot_filename)
plt.show()

# Plot all units for Sensor 2 (j=2) on the same plot
plt.figure(figsize=(10, 6))
for i in range(N):
    plt.plot(Y2_series[i].cpu().numpy(), label=f'Unit {i}')
plt.xlabel('Time')
plt.ylabel('Y_i,j(t)')
plt.title('Time Series for All Units (Sensor 2)')
plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1))
plt.grid(True)

# Save the plot
plot_filename = os.path.join(summary_plot_dir, 'all_units_sensor2.png')
plt.savefig(plot_filename)
plt.show()

# Plot Z1 and Z2 components
plt.figure(figsize=(12, 6))

# Z1 components
plt.subplot(1, 2, 1)
plt.plot(t, Z1_component1, label='$Z_{1,1}(t) = 1$')
plt.plot(t, Z1_component2, label='$Z_{1,2}(t) = t^{0.7} \cdot \sin(t)$')
plt.plot(t, Z1_component3, label='$Z_{1,3}(t) = t^2$')
plt.title('Components of $Z_1(t)$')
plt.xlabel('Time $t$')
plt.ylabel('Value')
plt.legend()
plt.grid(True)

# Z2 components
plt.subplot(1, 2, 2)
plt.plot(t, Z2_component1, label='$Z_{2,1}(t) = 1$')
plt.plot(t, Z2_component2, label='$Z_{2,2}(t) = t^{1.5}$')
plt.plot(t, Z2_component3, label='$Z_{2,3}(t) = t^2$')
plt.title('Components of $Z_2(t)$')
plt.xlabel('Time $t$')
plt.ylabel('Value')
plt.legend()
plt.grid(True)

plt.tight_layout()

# Save the plot
plot_filename = os.path.join(summary_plot_dir, 'Z_components.png')
plt.savefig(plot_filename)
plt.show()

print(f"Summary plots saved to {summary_plot_dir}")

# Define output directory
output_dir = 'fm2_data'
os.makedirs(output_dir, exist_ok=True)

# List to store data for all units
all_units_data = []

# Iterate through units and collect time series data
for i in range(N):
    unit_number = i + N + 1  # Assuming unit numbers are indexed from 1
    time_cycles = torch.cat((torch.arange(1, failure_times[i], device=device),
                             torch.tensor([failure_times[i]], device=device))).cpu().numpy()
    sensor_1_values = Y1_series[i].cpu().numpy()
    sensor_2_values = Y2_series[i].cpu().numpy()
    B_i1_values = B_i1_samples_list[i].cpu().numpy()
    B_i2_values = B_i2_samples_list[i].cpu().numpy()
    failure_time = failure_times_list[i].cpu().numpy()

    # Create DataFrame to match the format in the provided example
    df = pd.DataFrame({
        'failure mode': 2,  # Assuming failure mode is always 1 for this data
        'unit number': unit_number,
        'time, in cycles': time_cycles,
        'sensor 1': sensor_1_values,
        'sensor 2': sensor_2_values
    })

    # Append data to the list
    all_units_data.append(df)

# Concatenate all unit DataFrames into a single DataFrame
all_units_df = pd.concat(all_units_data, ignore_index=True)

# Save the combined DataFrame to CSV
csv_filename = os.path.join(output_dir, 'all_units_time_series_fm2.csv')
all_units_df.to_csv(csv_filename, index=False)

# Save B values and failure times to CSV
B_values_data = []
for i in range(N):
    unit_number = i + N + 1
    B_i1_values = B_i1_samples_list[i].cpu().numpy()
    B_i2_values = B_i2_samples_list[i].cpu().numpy()
    failure_time = failure_times_list[i].cpu().numpy()

    B_df = pd.DataFrame({
        'unit number': unit_number,
        'B_i1_1': B_i1_values[0],
        'B_i1_2': B_i1_values[1],
        'B_i1_3': B_i1_values[2],
        'B_i2_1': B_i2_values[0],
        'B_i2_2': B_i2_values[1],
        'B_i2_3': B_i2_values[2],
        'failure time': failure_time
    }, index=[0])
    B_values_data.append(B_df)

B_values_df = pd.concat(B_values_data, ignore_index=True)
B_csv_filename = os.path.join(output_dir, 'B_values_failure_times_fm2.csv')
B_values_df.to_csv(B_csv_filename, index=False)

# Create log file to save sigma, lambda, and alpha values
log_filename = os.path.join(output_dir, 'parameters_log_fm2.txt')
with open(log_filename, 'w') as log_file:
    log_file.write(f"sigma_j1: {sigma_j1}\n")
    log_file.write(f"sigma_j2: {sigma_j2}\n")
    # log_file.write(f"lambda: {lambda_}\n")
    # log_file.write(f"alpha: {alpha}\n")
    log_file.write(f"beta1: {beta_1}\n")
    log_file.write(f"beta2: {beta_2}\n")
    log_file.write(f"N: {N}\n")

print(f"CSV files and log file saved to {output_dir}")
