"""
This script performs Variational Inference for MFGPCox model.
It optimizes the Evidence Lower Bound (ELBO) to estimate failure probabilities
and survival times for engineering units based on sensor data.

Author: Sina Aghaee Dabaghan Fard
"""

# =============================================================================
# Imports (Preserved as requested)
# =============================================================================

from torch.special import digamma
import torch.optim as optim
from datetime import datetime
from utils.utils_final.plot_save_print import *
from utils.utils_final.data_processing import *
from utils.utils_final.CMGP import *
from utils.utils_final.options import *
from torch import lgamma
import neptune

import time
from pathlib import Path
import os
import pandas as pd
import torch


# =============================================================================
# Configuration & Utilities
# =============================================================================

class Config:
    """Configuration parameters for the optimization process."""

    # Hardware Configuration
    # Set to False to force CPU usage even if GPU is available
    USE_CUDA = False

    # Hyperparameters
    SEED = 423
    INDUCING_POINTS = 50
    NUM_ITERATIONS = 12000
    LEARNING_RATE = 0.01

    # Device Logic
    if USE_CUDA and torch.cuda.is_available():
        DEVICE = torch.device("cuda:0")
        print(f"[INFO] Running on {torch.cuda.get_device_name(0)}")
    else:
        DEVICE = torch.device("cpu")
        print("[INFO] Running on CPU")

    # Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_FILE = "historical_data.csv"

    # Sensor Configuration
    SENSORS = ["sensor 1", "sensor 2"]


def fmt_time(seconds: float) -> str:
    """Formats time duration into a readable string."""
    return f"{seconds:.6f} sec ({seconds / 60.0:.6f} min)"


def append_txt(path: str, lines: list) -> None:
    """Appends lines of text to a file, creating directories if needed."""
    Path(os.path.dirname(path) or ".").mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln + "\n")


# Apply Global Settings
device = Config.DEVICE
torch.manual_seed(Config.SEED)
if device.type == 'cuda':
    torch.cuda.manual_seed(Config.SEED)

# =============================================================================
# Data Loading & Preprocessing
# =============================================================================

print(f"[INFO] Loading data from: {Config.BASE_DIR}...")

# Load DataFrames
all_data_transformed = pd.read_csv(os.path.join(Config.BASE_DIR, Config.DATA_FILE))
train_data_transformed = pd.read_csv(os.path.join(Config.BASE_DIR, Config.DATA_FILE))

# Type casting
all_data_transformed['time, in cycles'] = all_data_transformed['time, in cycles'].astype('float32')
train_data_transformed['time, in cycles'] = train_data_transformed['time, in cycles'].astype('float32')

# Create Data Dictionaries
data_dicts, all_sensor_readings, all_time_points = create_data_dicts(all_data_transformed, Config.SENSORS)

failure_modes = all_data_transformed['failure mode'].unique()
sensors = all_data_transformed.columns.intersection(Config.SENSORS)
failure_modes_sensors = [(sensor, fm) for sensor in sensors for fm in failure_modes]

# =============================================================================
# Unit Processing & Event Times
# =============================================================================

train_units_event_time = {}
test_units_event_time = {}

unique_historical_units = train_data_transformed['unit number'].unique()
unique_units = all_data_transformed['unit number'].unique()
train_unit_range = unique_historical_units

# Process Event Times
for unit in unique_units:
    subset_df = all_data_transformed[all_data_transformed['unit number'] == unit]
    max_event_time = subset_df['time, in cycles'].max()
    event_time_tensor = torch.tensor(max_event_time, dtype=torch.float32, device=device)

    if unit in train_unit_range:
        train_units_event_time[unit] = event_time_tensor
    else:
        test_units_event_time[unit] = event_time_tensor

# Calculate Minimum Event Times (V) by Failure Mode
min_V_by_failure_mode = {}
failure_mode_groups = train_data_transformed.groupby('failure mode')

for failure_mode, group in failure_mode_groups:
    train_units_event_time_mode = {
        unit: train_units_event_time[unit]
        for unit in group['unit number'].unique()
        if unit in train_units_event_time
    }
    if train_units_event_time_mode:
        V = torch.tensor(list(train_units_event_time_mode.values()))
        min_V_by_failure_mode[failure_mode] = V.min()

# Initialize Unit Metadata
unit_status = {}
unit_manufacturer = {}
unit_failure_mode = {}

for unit in unique_units:
    subset_df = all_data_transformed[all_data_transformed['unit number'] == unit]
    unit_status[unit] = torch.tensor(1, dtype=torch.float32, device=device)
    unit_manufacturer[unit] = torch.tensor(0, dtype=torch.float32, device=device)
    unit_failure_mode[unit] = subset_df['failure mode'].max()

all_units = unique_units.tolist()
historical_units = unique_historical_units.tolist()
test_units = [unit for unit in all_units if unit not in historical_units]

# =============================================================================
# Hyperparameter Loading (CMGP)
# =============================================================================

sensor_gp_hyperparameter_paths = {
    'sensor 1': r"CMGP\sensor_1\optimized_CMGP_hyperparameters\optimized_CMGP_hyperparameters_unit_specific"
                r"\optimized_CMGP_hyperparameters_unit_specific.pth",
    'sensor 2': r"CMGP\sensor_2\optimized_CMGP_hyperparameters\optimized_CMGP_hyperparameters_unit_specific"
                r"\optimized_CMGP_hyperparameters_unit_specific.pth"
}

loaded_hyperparameters = {}
loaded_lambda_hyp = {}

for sensor, path in sensor_gp_hyperparameter_paths.items():
    full_path = os.path.join(Config.BASE_DIR, path)
    loaded_hyperparams = torch.load(full_path, map_location=device, weights_only=False)
    loaded_hyperparameters.update(loaded_hyperparams['optimized_hyperparameters'])

    optimized_lambda_hyp = loaded_hyperparams.get('optimized_lambda_hyp', {})
    for i in range(1, 4):
        if i in optimized_lambda_hyp:
            loaded_lambda_hyp[(sensor, i)] = optimized_lambda_hyp[i]

# =============================================================================
# Variational Parameter Initialization
# =============================================================================

# --- Gamma, Beta, Alpha Initializations ---
initial_gamma = {mode: torch.rand(1, device=device) for mode in failure_modes}
initial_alpha_hat = {mode: torch.rand(1, device=device) for mode in failure_modes}
initial_beta = {(sensor, fm): torch.rand(1, device=device) for (sensor, fm) in failure_modes_sensors}

# Prior Alpha
alpha0 = {mode: torch.rand(1, device=device) for mode in failure_modes}
alpha0[1] = torch.tensor(0.1, device=device)
alpha0[2] = torch.tensor(0.1, device=device)

# Manual overrides for Beta
initial_beta[('sensor 1', 1)] = torch.tensor(0.01, device=device)
initial_beta[('sensor 1', 2)] = torch.tensor(0.01, device=device)
initial_beta[('sensor 2', 1)] = torch.tensor(0.01, device=device)
initial_beta[('sensor 2', 2)] = torch.tensor(0.01, device=device)

# --- Prior Distributions for b and rho ---

# mu_b_0 = {1: torch.tensor(0.0, device=device), 2: torch.tensor(0.0, device=device)}
# sigma_b_0 = {1: torch.tensor(10.0, device=device), 2: torch.tensor(10.0, device=device)}
# alpha_rho_0 = {1: torch.tensor(1.0, device=device), 2: torch.tensor(1.0, device=device)}
# beta_rho_0 = {1: torch.tensor(0.1, device=device), 2: torch.tensor(0.1, device=device)}


mu_b_0 = {1: torch.tensor(0.0, device=device),
          2: torch.tensor(0.0, device=device)}
sigma_b_0 = {1: torch.tensor(10.0, device=device),
             2: torch.tensor(10.0, device=device)}
alpha_rho_0 = {1: torch.tensor(0.0002, device=device),
               2: torch.tensor(0.0002, device=device)}
beta_rho_0 = {1: torch.tensor(0.002, device=device),
              2: torch.tensor(0.002, device=device)}



# --- Variational Parameters (Hats) ---
initial_mu_b_hat = {mode: torch.rand(1, device=device) for mode in failure_modes}
initial_mu_b_hat[1] = torch.tensor(-6.0, device=device)
initial_mu_b_hat[2] = torch.tensor(-8.0, device=device)

initial_sigma_b_hat = {mode: torch.rand(1, device=device) for mode in failure_modes}
initial_sigma_b_hat[1] = torch.tensor(0.16, device=device)
initial_sigma_b_hat[2] = torch.tensor(0.16, device=device)

initial_alpha_rho_hat = {mode: torch.rand(1, device=device) for mode in failure_modes}
initial_alpha_rho_hat[1] = torch.log(torch.tensor(0.022 * 300, device=device))
initial_alpha_rho_hat[2] = torch.log(torch.tensor(0.07 * 300, device=device))

initial_beta_rho_hat = {mode: torch.rand(1, device=device) for mode in failure_modes}
initial_beta_rho_hat[1] = torch.log(torch.tensor(1.0 * 300, device=device))
initial_beta_rho_hat[2] = torch.log(torch.tensor(1.0 * 300, device=device))

# =============================================================================
# Covariance Approximation & Pre-computation
# =============================================================================

approx_cov_results = {}

for (sensor, fm) in failure_modes_sensors:
    approx_cov_matrix, a, d_plus_noise_matrix, d_plus_noise_matrix_inv, k_u_f_stacked, u_mean_m = (
        get_approximated_covariance_matrix(
            data_dicts.get((sensor, fm)),
            loaded_hyperparameters,
            loaded_lambda_hyp.get((sensor, fm)),
            128
        )
    )
    approx_cov_results[(sensor, fm)] = {
        'approx_cov_matrix': approx_cov_matrix,
        'a': a,
        'd_plus_noise_matrix': d_plus_noise_matrix,
        'd_plus_noise_matrix_inv': d_plus_noise_matrix_inv,
        'k_u_f_stacked': k_u_f_stacked,
        'u_mean_m': u_mean_m
    }

# Precompute GP Means for Historical Units
precomputed_means = {}
for unit in historical_units:
    unit_failure = unit_failure_mode.get(unit)
    unit_event_time = train_units_event_time.get(unit).unsqueeze(0)

    mean, _ = get_cmgp_predictions(
        all_sensor_readings, unit_event_time, 128, unit,
        Config.SENSORS, unit_failure, data_dicts, loaded_hyperparameters,
        loaded_lambda_hyp, approx_cov_results, preferred_device=device
    )
    precomputed_means[(unit, unit_failure)] = mean.squeeze()

# Precompute Predictions for Integrals
precomputed_predictions = {}


def precompute_predictions_for_integral(units, sensors, data, hyperparameters, lambda_hyperparameter):
    """
    Pre-computes mean and variance predictions for each unit along the time axis.
    Used for efficient numerical integration in Cox model.
    """
    for unit in units:
        unit_failure = unit_failure_mode.get(unit)
        unit_event_time = (train_units_event_time if unit in historical_units else test_units_event_time).get(unit)

        num_points = 1000
        ls = torch.linspace(0, unit_event_time, num_points).to(device)

        # Compute means and variances once for all l values
        means, variances = get_cmgp_predictions(
            all_sensor_readings, ls, 128, unit, sensors, unit_failure,
            data, hyperparameters, lambda_hyperparameter, approx_cov_results, preferred_device=device
        )

        precomputed_predictions[unit] = {
            'ls': ls,
            'means': means.squeeze(),
            'variances_diagonal': torch.diagonal(variances, dim1=1, dim2=2).squeeze()
        }


# Run precomputation
all_units_list = historical_units + test_units
precompute_predictions_for_integral(all_units_list, Config.SENSORS, data_dicts, loaded_hyperparameters,
                                    loaded_lambda_hyp)


# =============================================================================
# Mathematical Model Definition (ELBO Terms)
# =============================================================================

def h0(l, mu_b_hat, sigma_b_hat, alpha_rho_hat, beta_rho_hat, min_V):
    """
    Computes the baseline hazard function.

    Equation:
    $$ h_0(t) = \exp(\mu_b + \frac{\sigma_b^2}{2}) \cdot (1 - \frac{l}{\beta_\rho})^{-\alpha_\rho} $$
    """
    return torch.exp((mu_b_hat + ((sigma_b_hat ** 2) / 2))) * (1 - (l / beta_rho_hat)) ** (-alpha_rho_hat)


def get_cox1(data, sensor, mu_b_dict, alpha_rho_dict, beta_rho_dict, beta_dict, gamma_dict, hyperparameters,
             lambda_hyperparameter):
    """
    Calculates the first Cox partial likelihood term (linear component).
    """
    cox1 = torch.tensor(0.0, device=device).unsqueeze(-1)

    for unit in historical_units:
        unit_failure = unit_failure_mode.get(unit)
        mu_b = mu_b_dict.get(unit_failure)
        alpha_rho = alpha_rho_dict.get(unit_failure)
        beta_rho = beta_rho_dict.get(unit_failure)

        # Gather Beta coefficients
        beta_values = []
        for s in Config.SENSORS:
            beta_val = beta_dict.get((s, unit_failure))
            if beta_val is not None:
                beta_values.append(beta_val)
            else:
                raise ValueError(f"Beta value for sensor {s} and failure mode {unit_failure} not found.")

        beta = torch.stack(beta_values).to(device)
        gamma = gamma_dict.get(unit_failure)  # we don"t have X in NS & CS

        status = unit_status.get(unit)
        unit_event_time = train_units_event_time.get(unit)

        first_term = (mu_b + (alpha_rho / beta_rho) * unit_event_time)

        # Retrieve precomputed GP mean
        mean = precomputed_means[(unit, unit_failure)]

        if beta.shape != mean.shape:
            raise ValueError(f"Shape mismatch: beta {beta.shape} vs mean {mean.shape} for unit {unit}.")

        third_term = beta @ mean
        cox1 += status * (first_term + third_term)

    return cox1


def integrand(precomputed, mu_b_hat, sigma_b_hat, alpha_rho_hat, beta_rho_hat, beta, gamma, unit):
    """
    Computes the integrand for the Cox integral approximation.
    """
    h0_value = h0(
        precomputed['ls'],
        mu_b_hat, sigma_b_hat, alpha_rho_hat, beta_rho_hat,
        min_V_by_failure_mode.get(unit_failure_mode.get(unit))
    )

    exp_component_first = beta @ precomputed['means']
    exp_component_second = 0.5 * beta ** 2 @ precomputed['variances_diagonal']

    exp_component = exp_component_first + exp_component_second

    return h0_value * torch.exp(exp_component)


def get_cox2(sensors, mu_b_dict, sigma_b_dict, alpha_rho_dict, beta_rho_dict, beta_dict, gamma_dict,
             precomputed_predictions):
    """
    Calculates the second Cox term (integral component) using Trapezoidal integration.
    """
    cox2 = torch.tensor(0.0, device=device).unsqueeze(-1)

    for unit in historical_units + test_units:
        unit_failure = unit_failure_mode.get(unit)

        mu_b = mu_b_dict.get(unit_failure)
        sigma_b = sigma_b_dict.get(unit_failure)
        alpha_rho = alpha_rho_dict.get(unit_failure)
        beta_rho = beta_rho_dict.get(unit_failure)

        beta_values = [beta_dict.get((s, unit_failure)) for s in sensors]
        beta = torch.stack(beta_values).to(device)
        gamma = gamma_dict.get(unit_failure)

        precomputed = precomputed_predictions[unit]
        vals = integrand(precomputed, mu_b, sigma_b, alpha_rho, beta_rho, beta, gamma, unit)

        integral_approx = torch.trapz(vals, precomputed['ls'])
        cox2 += integral_approx

    return cox2


def kl_term(data, alpha_hat_dict):
    """
    Computes the Kullback-Leibler (KL) divergence term for Dirichlet/Gamma distributions.
    """
    alpha_hat_vector = torch.stack(list(alpha_hat_dict.values())).to(device)
    alpha0_vector = torch.stack(list(alpha0.values())).to(device)

    summation_first = torch.lgamma(torch.sum(alpha_hat_vector)) - torch.lgamma(torch.sum(alpha0_vector))
    summation_second = torch.sum(torch.lgamma(alpha0_vector) - torch.lgamma(alpha_hat_vector))

    digamma_diff = torch.digamma(alpha_hat_vector) - torch.digamma(torch.sum(alpha_hat_vector))
    summation_third = torch.sum((alpha_hat_vector - alpha0_vector) * digamma_diff)

    negative_kl = -(summation_first + summation_second + summation_third)

    return negative_kl + torch.sum(digamma_diff)


def kl_b_rho(mu_b_0_dict, sigma_b_0_dict, alpha_rho_0_dict, beta_rho_0_dict, mu_b_hat_dict,
             sigma_b_hat_dict, alpha_rho_hat_dict, beta_rho_hat_dict):
    """
    Computes KL divergence for parameters b (Normal) and rho (Gamma).

    $$ KL(q(b)||p(b)) + KL(q(\rho)||p(\rho)) $$
    """
    total_kl = torch.tensor(0.0, device=device).unsqueeze(-1)

    for failure_mode in failure_modes:
        # Retrieve priors
        mu_b, sigma_b = mu_b_0_dict[failure_mode], sigma_b_0_dict[failure_mode]
        alpha_rho, beta_rho = alpha_rho_0_dict[failure_mode], beta_rho_0_dict[failure_mode]

        # Retrieve variational params
        mu_b_hat, sigma_b_hat = mu_b_hat_dict[failure_mode], sigma_b_hat_dict[failure_mode]
        alpha_rho_hat, beta_rho_hat = alpha_rho_hat_dict[failure_mode], beta_rho_hat_dict[failure_mode]

        # KL for Normal (b)
        kl_b = (0.5) * (torch.log((sigma_b / sigma_b_hat) ** 2) +
                        ((sigma_b_hat ** 2 + (mu_b_hat - mu_b) ** 2) / (sigma_b ** 2)))

        # KL for Gamma (rho)
        kl_rho = ((alpha_rho_hat * torch.log(beta_rho_hat) - alpha_rho * torch.log(beta_rho))
                  - (lgamma(alpha_rho_hat) - lgamma(alpha_rho))
                  + (alpha_rho_hat - alpha_rho) * (digamma(alpha_rho_hat) - torch.log(beta_rho_hat))
                  - (beta_rho_hat - beta_rho) * (alpha_rho_hat / beta_rho_hat))

        total_kl += kl_b + kl_rho

    return total_kl


def negative_elbo(sensor, mu_b_0_dict, sigma_b_0_dict, alpha_rho_0_dict, beta_rho_0_dict, mu_b_hat_dict,
                  sigma_b_hat_dict, alpha_rho_hat_dict, beta_rho_hat_dict, beta_dict, gamma_dict, alpha_dict,
                  data, hyperparameters, lambda_hyp):
    """
    Computes the Negative Evidence Lower Bound (ELBO) to be minimized.

    $$ \mathcal{L} = \mathbb{E}[\log p] - KL(q||p) $$
    """
    cox_1 = get_cox1(data, sensor, mu_b_hat_dict, alpha_rho_hat_dict, beta_rho_hat_dict, beta_dict, gamma_dict,
                     hyperparameters, lambda_hyp)

    cox_2 = get_cox2(Config.SENSORS, mu_b_hat_dict, sigma_b_hat_dict, alpha_rho_hat_dict, beta_rho_hat_dict, beta_dict,
                     gamma_dict, precomputed_predictions)

    kl = kl_term(data, alpha_dict)

    kl_of_b_and_rho = kl_b_rho(mu_b_0_dict, sigma_b_0_dict, alpha_rho_0_dict, beta_rho_0_dict, mu_b_hat_dict,
                               sigma_b_hat_dict, alpha_rho_hat_dict, beta_rho_hat_dict)

    # Returning individual components for logging purposes
    total_loss = -(cox_1 - cox_2 + kl - kl_of_b_and_rho)
    return total_loss, -(cox_1 - cox_2), -kl, kl_of_b_and_rho


def objective_function(sensor, mu_b_0_dict, sigma_b_0_dict, alpha_rho_0_dict, beta_rho_0_dict, flat_parameters,
                       metadata, data, hyperparameters, lambda_hyp):
    """Reconstructs parameters and computes the negative ELBO."""

    reconstructed_params = reconstruct_hyperparameters_elbo_2(flat_parameters, metadata)

    neg_elbo, cox_loss, kl_loss, kl_b_rho_loss = negative_elbo(
        sensor, mu_b_0_dict, sigma_b_0_dict, alpha_rho_0_dict, beta_rho_0_dict,
        reconstructed_params['mu_b_hat'],
        reconstructed_params['sigma_b_hat'],
        reconstructed_params['alpha_rho_hat'],
        reconstructed_params['beta_rho_hat'],
        reconstructed_params['beta'],
        reconstructed_params['gamma'],
        reconstructed_params['alpha_hat'],
        data, hyperparameters, lambda_hyp
    )

    return neg_elbo, cox_loss, kl_loss, kl_b_rho_loss


# =============================================================================
# Main Optimization Loop
# =============================================================================

if __name__ == "__main__":

    # Initialize Optimizer
    flat_parameters_initial, metadata = flatten_hyperparameters_elbo_3(
        initial_mu_b_hat, initial_sigma_b_hat, initial_alpha_rho_hat,
        initial_beta_rho_hat, initial_alpha_hat, initial_beta, initial_gamma
    )

    flat_parameters_initial.requires_grad_(True)
    optimizer = optim.Adam([flat_parameters_initial], lr=Config.LEARNING_RATE)
    losses = []


    def closure():
        """Optimizer closure for calculating loss and gradients."""
        optimizer.zero_grad()
        total_loss, cox_loss, kl_loss, kl_b_rho_loss = objective_function(
            Config.SENSORS, mu_b_0, sigma_b_0, alpha_rho_0, beta_rho_0,
            flat_parameters_initial, metadata, data_dicts, loaded_hyperparameters, loaded_lambda_hyp
        )
        total_loss.backward()
        losses.append((total_loss.item(), cox_loss.item(), kl_loss.item(), kl_b_rho_loss.item()))
        return total_loss


    # Setup Logging
    start_time = time.time()
    header_written = False

    formatted_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"optimized_parameters_unit_2S_final_Revision_time_tracking"
    main_folder_path = os.path.join(Config.BASE_DIR, f"{file_name}_{formatted_now}")
    os.makedirs(main_folder_path, exist_ok=True)

    print("[INFO] Starting Optimization...")
    print(f"[INFO] Saving results to: {main_folder_path}")

    # Training Loop
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    train_t0 = time.perf_counter()

    for iteration in range(Config.NUM_ITERATIONS):
        optimizer.step(closure)

        header_written = save_and_print_parameters_3(
            iteration=iteration,
            num_iterations=Config.NUM_ITERATIONS,
            losses=losses,
            flat_parameters_initial=flat_parameters_initial,
            metadata=metadata,
            main_folder_path=main_folder_path,
            file_name=file_name,
            save_interval=5000,
            print_interval=100,
            header_written=header_written,
            start_time=start_time,
        )

    # Timing Summary
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    train_sec = time.perf_counter() - train_t0

    timing_lines = [
        "===== TRAINING TIME SUMMARY =====",
        f"num_iterations: {Config.NUM_ITERATIONS}",
        f"total_training_time: {fmt_time(train_sec)}",
        "================================",
        ""
    ]

    print("\n".join(timing_lines))
    append_txt(os.path.join(main_folder_path, "timing_total.txt"), timing_lines)

    print("[INFO] Optimization Complete.")
