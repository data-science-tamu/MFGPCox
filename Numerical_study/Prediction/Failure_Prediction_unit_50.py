import torch
import numpy as np
from utils.utils_final_fm.plot_save_print_fm import *
from utils.utils_final_fm.data_processing_fm import *
from utils.utils_final_fm.CMGP_fm import *
from utils.utils_final_fm.rul_pred_fm import *
from utils.utils_final_fm.options_fm import *
import csv
import logging
import time
from sklearn.metrics import roc_auc_score, roc_curve

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

########################################################################################################################
base_dir = os.path.dirname(os.path.abspath(__file__))
########################################################################################################################
NS = '19'
t = 50
NN_loss_type = 2

all_data_transformed = pd.read_csv(os.path.join(base_dir, f"historical_plus_test_data_{t}.csv"))

train_data_transformed = pd.read_csv(os.path.join(base_dir, "historical_data.csv"))

all_data_transformed['time, in cycles'] = all_data_transformed['time, in cycles'].astype('float32')
train_data_transformed['time, in cycles'] = train_data_transformed['time, in cycles'].astype('float32')

sensors_list = ["sensor 1", "sensor 2"]

sensors_str = "_".join(sensors_list)

save_directory = f'{sensors_str}_t_{t}_NNloss{NN_loss_type}_c-fm_sp-ind_cprob_median_c_elbo'

# if os.path.exists(save_directory):
#     raise FileExistsError(f"The directory '{save_directory}' already exists.")
# else:
os.makedirs(save_directory,exist_ok=True)
print(f"The directory '{save_directory}' has been created.")

log_filename = os.path.join(save_directory, f'log_{sensors_str}_t{t}_fm_exact.txt')

# Configure logging
logging.basicConfig(
    filename=log_filename,
    filemode='w',
    level=logging.INFO,
    format='%(message)s'
)

log_message(f'num_of_sensors: {len(sensors_list)} ==> sensors: {sensors_list}\n')

data_dicts, all_sensor_readings, all_time_points = create_data_dicts(all_data_transformed, sensors_list)

failure_modes = all_data_transformed['failure mode'].unique()

sensors = all_data_transformed.columns.intersection(sensors_list)

# Generate a list of (sensor, failure_mode) combinations
failure_modes_sensors = [(sensor, fm) for sensor in sensors for fm in failure_modes]

(train_units_event_time, test_units_event_time, min_V_by_failure_mode, unit_status, unit_manufacturer,
 unit_failure_mode) = process_unit_data(train_data_transformed, all_data_transformed, device)
########################################################################################################################
test_data_transformed = pd.read_csv(os.path.join(base_dir, f"test_data_fm_{t}.csv"))
test_data_transformed['time, in cycles'] = test_data_transformed['time, in cycles'].astype('float32')

data_dicts_tests, _, _ = create_data_dicts(test_data_transformed, sensors_list)

(_, _, _, _, _,
 unit_failure_mode) = process_unit_data(test_data_transformed, test_data_transformed, device)
########################################################################################################################
# loading gp hyperparameters

loaded_hyperparameters = {}
loaded_lambda_hyp = {}

sensor_gp_hyperparameter_paths = {
    'sensor 1': "./sensor_1/optimized_hyperparameters_fm_specific_2024-12-25_19-01-31/iteration_50000"
                "/optimized_hyperparameters_fm_specific_50000.pth",

    'sensor 2': "./sensor_2/optimized_hyperparameters_fm_specific_2024-12-25_19-06-11/iteration_20000"
                "/optimized_hyperparameters_fm_specific_20000.pth"
}


for sensor, path in sensor_gp_hyperparameter_paths.items():
    loaded_hyperparams = torch.load(path, map_location=device)
    loaded_hyperparameters.update(loaded_hyperparams['optimized_hyperparameters'])
    optimized_lambda_hyp = loaded_hyperparams.get('optimized_lambda_hyp', )

    for i in range(1, 4):
        if i in optimized_lambda_hyp:
            loaded_lambda_hyp[(sensor, i)] = optimized_lambda_hyp[i]

#######################################################################################################################
# loading Cox parameters

loaded_params = torch.load(os.path.join(base_dir, r"optimized_parameters_unit_2S_2025-05-15_18-25-48/iteration_33000"
                                                  r"/optimized_parameters_unit_2S_final_cottected_pi_33000.pth"),
                           map_location=device)


# loaded_params = torch.load(os.path.join(base_dir, r"optimized_parameters_unit_2S_final_weight_on_kl_2025-01-04_21-57"
#                                                   r"-39/iteration_15000"
#                                                   r"/optimized_parameters_unit_2S_final_weight_on_kl_15000.pth"))

# loaded_pi_hat = loaded_params['optimized_pi_hat']
loaded_mu_b_hat = loaded_params['optimized_mu_b_hat']
loaded_sigma_b_hat = loaded_params['optimized_sigma_b_hat']
loaded_alpha_rho_hat = loaded_params['optimized_alpha_rho_hat']
loaded_beta_rho_hat = loaded_params['optimized_beta_rho_hat']
loaded_alpha_hat = loaded_params['optimized_alpha_hat']
loaded_gamma = loaded_params['optimized_gamma']
loaded_beta = loaded_params['optimized_beta']

loaded_pi_hat = {
    1: torch.tensor(0.5, device=device),
    2: torch.tensor(0.5, device=device)
}

########################################################################################################################
log_message(
    f"beta:{loaded_beta},\nmu_b_hat:{loaded_mu_b_hat},\nsigma_b_hat:{loaded_sigma_b_hat},\nalpha_rho_hat:{loaded_alpha_rho_hat},\nbeta_rho_hat:{loaded_beta_rho_hat},\nalpha_hat:{loaded_alpha_hat},\npi_hat:{loaded_pi_hat},"
    f"\ngamma:{loaded_gamma}\n")
########################################################################################################################

approx_cov_results = {}

for (sensor, fm) in failure_modes_sensors:
    approx_cov_matrix, a, d_plus_noise_matrix, d_plus_noise_matrix_inv, k_u_f_stacked, u_mean_m = (
        get_approximated_covariance_matrix(
            data_dicts.get((sensor, fm)),
            loaded_hyperparameters,
            loaded_lambda_hyp.get((sensor, fm)),
            inducing_points_num))

    approx_cov_results[(sensor, fm)] = {
        'approx_cov_matrix': approx_cov_matrix,
        'a': a,
        'd_plus_noise_matrix': d_plus_noise_matrix,
        'd_plus_noise_matrix_inv': d_plus_noise_matrix_inv,
        'k_u_f_stacked': k_u_f_stacked,
        'u_mean_m': u_mean_m
    }

########################################################################################################################
# probability of failure

# unit_f = [(11, 1), (12, 1), (13, 1), (14, 1), (15, 1),
#           (41, 2), (42, 2), (43, 2), (44, 2), (45, 2),
#           (111, 3), (112, 3), (113, 3), (114, 3), (115, 3), ]


unit_f = test_data_transformed[['unit number', 'failure mode']].drop_duplicates().sort_values(by=['unit number'])
unit_f = list(unit_f.itertuples(index=False, name=None))

start_time = time.time()

unit_f_probabilities = {}

pi1 = loaded_pi_hat.get(1)
pi2 = loaded_pi_hat.get(2)

df_results = pd.DataFrame(columns=['Unit', 'P1', 'P2'])

for unit in unit_f:
    unit_probs = []

    # Loop through each sensor for the current unit
    for sensor in sensors_list:
        # Call your function to get probabilities for each failure mode
        prob1, prob2 = likelihood_of_failure_all_test_ns(data_dicts_tests, data_dicts, all_sensor_readings,
                                                         loaded_hyperparameters, loaded_lambda_hyp,
                                                         approx_cov_results, unit[0], unit[1], loaded_pi_hat,
                                                         sensor)

        unit_probs.append((prob1, prob2))

    if unit[1] == 1:
        constant = 1
    else:
        constant = 1
    if unit[0] == 53:
        constant = 1e10
    # Combine probabilities for each failure mode across all sensors for this unit
    combined_prob1 = torch.prod(torch.tensor([prob1 * constant for prob1, _ in unit_probs]))
    combined_prob2 = torch.prod(torch.tensor([prob2 * constant for _, prob2 in unit_probs]))

    # Sum of combined probabilities for normalization
    total_prob = pi1 * combined_prob1 + pi2 * combined_prob2

    # Normalize to get final probabilities for each failure mode
    final_prob1 = combined_prob1 * pi1 / total_prob
    final_prob2 = combined_prob2 * pi2 / total_prob

    # Store the results
    unit_f_probabilities[unit[0]] = {1: final_prob1, 2: final_prob2}
    log_message(f'Unit:{unit[0]} ===> P1: {final_prob1.item()},  P2: {final_prob2.item()}')

    # Append the results to the DataFrame
    df_results = df_results._append({'Unit': int(unit[0]),
                                     'P1': final_prob1.item(),
                                     'P2': final_prob2.item()},
                                    ignore_index=True)

log_message('\n')

actual_failure_modes = {
    **{unit: 1 for unit in range(51, 61)},
    **{unit: 2 for unit in range(61, 121)}}

for unit, probabilities in unit_f_probabilities.items():
    # Find the failure mode with the highest probability
    failure_mode = max(probabilities, key=probabilities.get)

    log_message(f"Unit: {unit} ==> Failure Mode: {failure_mode}")

    # Update the Failure_Mode in the DataFrame
    df_results.loc[df_results['Unit'] == unit, 'Predicted_Failure_Mode'] = failure_mode

for unit, value in actual_failure_modes.items():
    df_results.loc[df_results['Unit'] == unit, 'Actual_Failure_Mode'] = value

log_message('\n')

# Save the DataFrame to a CSV file
df_results.to_csv(os.path.join(save_directory, f'failure_mode_probs_{t}.csv'), index=False)

log_message('Results saved to failure_mode_probs.csv')

predicted_failure_modes = {}

# Initialize counters
correct_predictions = 0
total_predictions = 0

# Counters for each failure mode
correct_per_mode = {1: 0, 2: 0}
total_per_mode = {1: 0, 2: 0}

# Calculate accuracy
for unit, probabilities in unit_f_probabilities.items():
    # Predicted failure mode is the mode with the highest probability
    predicted_failure_mode = max(probabilities, key=probabilities.get)
    predicted_failure_modes[unit] = predicted_failure_mode

    # Get the actual failure mode for the unit
    actual_failure_mode = actual_failure_modes.get(unit)

    # Update overall accuracy counters
    total_predictions += 1
    if predicted_failure_mode == actual_failure_mode:
        correct_predictions += 1

    # Update per failure mode accuracy counters
    if actual_failure_mode in correct_per_mode:
        total_per_mode[actual_failure_mode] += 1
        if predicted_failure_mode == actual_failure_mode:
            correct_per_mode[actual_failure_mode] += 1

# Calculate overall accuracy
overall_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

# Calculate accuracy per failure mode
accuracy_per_mode = {mode: correct_per_mode[mode] / total_per_mode[mode] if total_per_mode[mode] > 0 else 0
                     for mode in correct_per_mode}

# Print the results
log_message(f"Overall Accuracy: {overall_accuracy:.2f}")

log_message('\n')

for mode, accuracy in accuracy_per_mode.items():
    log_message(f"Accuracy for Failure Mode {mode}: {accuracy:.2f}")

# # Define unit_f_probabilities dictionary
# unit_f_probabilities = {
#     **{i: {1: torch.tensor(1.0), 2: torch.tensor(0.0)} for i in range(51, 61)},
#     **{i: {1: torch.tensor(0.0), 2: torch.tensor(1.0)} for i in range(111, 121)},
# }
#
# predicted_failure_modes = actual_failure_modes

########################################################################################################################
from utils.utils_final.plot_save_print import *
from utils.utils_final.data_processing import *
from utils.utils_final.CMGP import *
from utils.utils_final.rul_pred_unit import *
from utils.utils_final.options import *
import csv

all_data_transformed = pd.read_csv(os.path.join(base_dir, f"historical_plus_test_data_{t}.csv"))

train_data_transformed = pd.read_csv(os.path.join(base_dir, "historical_data.csv"))

all_data_transformed['time, in cycles'] = all_data_transformed['time, in cycles'].astype('float32')
train_data_transformed['time, in cycles'] = train_data_transformed['time, in cycles'].astype('float32')

data_dicts, all_sensor_readings, all_time_points = create_data_dicts(all_data_transformed, sensors_list)

failure_modes = all_data_transformed['failure mode'].unique()

sensors = all_data_transformed.columns.intersection(sensors_list)

failure_modes_sensors = [(sensor, fm) for sensor in sensors for fm in failure_modes]

(train_units_event_time, test_units_event_time, min_V_by_failure_mode, unit_status, unit_manufacturer,
 unit_failure_mode) = process_unit_data(train_data_transformed, all_data_transformed, device)
########################################################################################################################
test_data_transformed = pd.read_csv(os.path.join(base_dir, f"test_data_{t}.csv"))
test_data_transformed['time, in cycles'] = test_data_transformed['time, in cycles'].astype('float32')

data_dicts_tests, _, _ = create_data_dicts(test_data_transformed, sensors_list)

(_, _, _, _, _, unit_failure_mode) = process_unit_data(test_data_transformed, test_data_transformed, device)

# loading gp hyperparameters

loaded_hyperparameters = {}
loaded_lambda_hyp = {}

if t == 10:
    sensor_gp_hyperparameter_paths = {
        'sensor 1': "./sensor_1/optim_test_hyperparams_10_final_2025-01-03_13-54-41/iteration_40000"
                    "/optim_test_hyperparams_10_final_40000.pth",

        'sensor 2': "./sensor_2/optim_test_hyperparams_10_final_2025-01-03_13-57-12/iteration_40000"
                    "/optim_test_hyperparams_10_final_40000.pth"
    }

if t == 20:
    sensor_gp_hyperparameter_paths = {
        'sensor 1': "./sensor_1/optim_test_hyperparams_20_final_2025-01-03_14-09-20/iteration_40000"
                    "/optim_test_hyperparams_20_final_40000.pth",

        'sensor 2': "./sensor_2/optim_test_hyperparams_20_final_2025-01-03_14-19-20/iteration_40000"
                    "/optim_test_hyperparams_20_final_40000.pth"
    }

if t == 50:
    sensor_gp_hyperparameter_paths = {
        'sensor 1': "./sensor_1/optim_test_hyperparams_50_final_2025-01-04_09-08-59/iteration_40000"
                    "/optim_test_hyperparams_50_final_40000.pth",

        'sensor 2': "./sensor_2/optim_test_hyperparams_50_final_2025-01-04_09-10-29/iteration_40000"
                    "/optim_test_hyperparams_50_final_40000.pth"
    }


for sensor, path in sensor_gp_hyperparameter_paths.items():
    loaded_hyperparams = torch.load(path, map_location=device)
    loaded_hyperparameters.update(loaded_hyperparams['optimized_hyperparameters'])
    optimized_lambda_hyp = loaded_hyperparams.get('optimized_lambda_hyp', )

    for i in range(1, 4):
        if i in optimized_lambda_hyp:
            loaded_lambda_hyp[(sensor, i)] = optimized_lambda_hyp[i]



# loaded_hyperparameters[(54, 'sensor 2', 1)]['alpha'] = loaded_hyperparameters[(53, 'sensor 2', 1)]['alpha']
# loaded_hyperparameters[(54, 'sensor 2', 1)]['xi'] = loaded_hyperparameters[(53, 'sensor 2', 1)]['xi']

approx_cov_results = {}

for (sensor, fm) in failure_modes_sensors:
    approx_cov_matrix, a, d_plus_noise_matrix, d_plus_noise_matrix_inv, k_u_f_stacked, u_mean_m = (
        get_approximated_covariance_matrix(
            data_dicts.get((sensor, fm)),
            loaded_hyperparameters,
            loaded_lambda_hyp.get((sensor, fm)),
            inducing_points_num))

    approx_cov_results[(sensor, fm)] = {
        'approx_cov_matrix': approx_cov_matrix,
        'a': a,
        'd_plus_noise_matrix': d_plus_noise_matrix,
        'd_plus_noise_matrix_inv': d_plus_noise_matrix_inv,
        'k_u_f_stacked': k_u_f_stacked,
        'u_mean_m': u_mean_m
    }

########################################################################################################################

file_path_fm1 = ('./fm1_data/B_values_failure_times_fm1.csv')
file_path_fm2 = ('./fm2_data/B_values_failure_times_fm2.csv')

df_fm1 = pd.read_csv(file_path_fm1)
df_fm2 = pd.read_csv(file_path_fm2)
# df_fm2['unit number'] += 60

units_range_1 = list(range(51, 61))
units_range_2 = list(range(111, 121))

filtered_fm1 = df_fm1[df_fm1['unit number'].isin(units_range_1 + units_range_2)]
filtered_fm2 = df_fm2[df_fm2['unit number'].isin(units_range_1 + units_range_2)]

filtered_data = pd.concat([filtered_fm1, filtered_fm2])

current_time = t
unit_t = [(int(row['unit number']), current_time, row['failure time'] - current_time) for _, row in
          filtered_data.iterrows()]

# Display the result
log_message(unit_t)
########################################################################################################################

b_values_fm1 = pd.read_csv(file_path_fm1)
b_values_fm2 = pd.read_csv(file_path_fm2)


# b_values_fm2['unit number'] += 60
# t_max = 200


def get_B_values(unit_name, failure_mode):
    if failure_mode == 1:
        return b_values_fm1[b_values_fm1['unit number'] == unit_name]
    elif failure_mode == 2:
        return b_values_fm2[b_values_fm2['unit number'] == unit_name]
    else:
        raise ValueError(f"Unknown failure mode: {failure_mode}")


def Z1(t):
    return torch.stack([
        torch.ones_like(t),
        0.3 * (t ** 0.6 * torch.sin(t)),
        t ** 2
    ], dim=-1)


def Z2(t):
    return torch.stack([torch.ones_like(t), t ** 1.5, t ** 2], dim=-1)  # Shape: [T, 3]


def Z3(t):
    return torch.stack([
        torch.ones_like(t),
        t ** 0.8 * torch.cos(t),
        t ** 2 + t
    ], dim=-1)  # Shape: [T, 3]


def Z4(t):
    return torch.stack([torch.ones_like(t), t ** 1.5, t ** 2], dim=-1)  # Shape: [T, 3]


def get_baseline_hazard(t, b, rho):
    return torch.exp(b + rho * t)  # Shape: [T]


def true_hazard(t, B_i1_samples, B_i2_samples, failure_mode):
    if failure_mode == 1:
        Z1_t = Z1(t)  # Shape: [T, 3]
        Z2_t = Z2(t)  # Shape: [T, 3]
        beta_1, beta_2 = 0.02, 0.015
        b, rho = -7, 0.015
    elif failure_mode == 2:
        Z1_t = Z3(t)  # Shape: [T, 3]
        Z2_t = Z4(t)  # Shape: [T, 3]
        beta_1, beta_2 = 0.018, 0.01
        b, rho = -10, 0.0095
    else:
        raise ValueError(f"Unknown failure mode: {failure_mode}")

    Z1_t = Z1_t.unsqueeze(0)  # Shape: [1, T, 3]
    Z2_t = Z2_t.unsqueeze(0)  # Shape: [1, T, 3]

    Z1_t = Z1_t.expand(B_i1_samples.shape[0], -1, -1)  # Shape: [N, T, 3]
    Z2_t = Z2_t.expand(B_i2_samples.shape[0], -1, -1)  # Shape: [N, T, 3]

    ZT1_Bi1 = torch.matmul(Z1_t, B_i1_samples.unsqueeze(-1)).squeeze(-1)  # Shape: [N, T]
    ZT2_Bi2 = torch.matmul(Z2_t, B_i2_samples.unsqueeze(-1)).squeeze(-1)  # Shape: [N, T]

    baseline_hazard = get_baseline_hazard(t, b, rho)  # Shape: [T]
    baseline_hazard = baseline_hazard.unsqueeze(0).expand(B_i1_samples.shape[0], -1)  # Shape: [N, T]

    hazard = baseline_hazard * torch.exp((beta_1 * ZT1_Bi1) + (beta_2 * ZT2_Bi2))  # Shape: [N, T]

    return hazard


def true_survival(t, B_i1, B_i2, t_star, failure_mode, num_points=1000):
    ls = torch.linspace(t_star, t.squeeze(0), num_points).to(device)
    vals = true_hazard(ls, B_i1, B_i2, failure_mode)
    integral_approx = torch.trapz(vals, ls)
    survival = torch.exp(-integral_approx)
    return survival  # Shape: [N]


number_of_samples = 10000
log_message(f'number of samples: {number_of_samples}')
plots_directory = os.path.join(save_directory, f"plots_{t}_exact_ns{number_of_samples}")
survivals_directory = os.path.join(save_directory, f"sp_{t}_exact_ns{number_of_samples}")

survival_plot_dir = os.path.join(plots_directory, 'survival')
box_plot_dir = os.path.join(plots_directory, 'boxplot')
sp_prob_dir = os.path.join(save_directory, 'sp_NN_ideal_500')

os.makedirs(sp_prob_dir, exist_ok=True)
os.makedirs(survival_plot_dir, exist_ok=True)
os.makedirs(box_plot_dir, exist_ok=True)
# os.makedirs(survivals_directory, exist_ok=True)
os.makedirs(survivals_directory, exist_ok=True)
os.makedirs(plots_directory, exist_ok=True)

absolute_errors = []
unit_mae = {}


for unit in unit_t:
    t_star = unit[1]
    unit_name = unit[0]
    # if unit_name not in [55, 59]:
    #     continue
    print(unit_name)
    actual_failure = unit[1] + unit[2]
    failure_mode = actual_failure_modes.get(unit_name)
    t_max = 191

    B_values = get_B_values(unit_name, failure_mode)
    B_i1_samples = torch.tensor(B_values[['B_i1_1', 'B_i1_2', 'B_i1_3']].values, device=device, dtype=torch.float32)
    B_i2_samples = torch.tensor(B_values[['B_i2_1', 'B_i2_2', 'B_i2_3']].values, device=device, dtype=torch.float32)

    survival = []
    failure = []
    predicted_survival = []
    abs_errors_for_unit = []
    survival_means = []
    survival_lower_bounds = []
    survival_upper_bounds = []

    file_path = os.path.join(survivals_directory, f'{unit_name}.tsv')
    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerow(
            ['Time', 'Failure_Probability', 'Survival_Probability', 'lower_bound', 'Upper_bound',
             'True_Survival_Probability', 'Actual_Failure'])

        for time in range(t_star, t_max):
            print(time)
            survival_distribution = St_cond_EST_prob_batch(min_V_by_failure_mode, unit_f_probabilities, failure_modes,
                                                           all_sensor_readings, approx_cov_results, unit_manufacturer,
                                                           sensors, unit_name, t_star, time, loaded_mu_b_hat,
                                                           loaded_sigma_b_hat, loaded_alpha_rho_hat,
                                                           loaded_beta_rho_hat, loaded_beta, loaded_gamma,
                                                           data_dicts, loaded_hyperparameters, loaded_lambda_hyp,
                                                           number_of_samples)

            # mean_survival = torch.mean(survival_distribution).item()
            mean_survival = torch.median(survival_distribution).item()
            lower_bound = torch.quantile(survival_distribution, 0.025).item()
            upper_bound = torch.quantile(survival_distribution, 0.975).item()
            survival_means.append(mean_survival)
            survival_lower_bounds.append(lower_bound)
            survival_upper_bounds.append(upper_bound)

            failure_probability = 1 - mean_survival
            predicted_survival.append(mean_survival)

            true_survival_prob = true_survival(torch.tensor([time], device=device), B_i1_samples, B_i2_samples,
                                               t_star, failure_mode)
            survival.append(true_survival_prob.cpu())
            failure.append(failure_probability)

            abs_error = abs(mean_survival - true_survival_prob.cpu().item())
            abs_errors_for_unit.append(abs_error)

            writer.writerow(
                [time, failure_probability, mean_survival, lower_bound, upper_bound, true_survival_prob.item(),
                 actual_failure])

    absolute_errors.append(abs_errors_for_unit)
    unit_mae[unit_name] = np.mean(abs_errors_for_unit)

    mean_bounds_dict = {}

    datasets = {
        "s_NN": {
            "fm1_path": f"./NN_model_NS_{NS}_FM1_ideal_2/survivals_probs_loss{NN_loss_type}_t{t}_ndata50/sp_{unit_name}.npy",
            "fm2_path": f"./NN_model_NS_{NS}_FM2_ideal_2/survivals_probs_loss{NN_loss_type}_t{t}_ndata50/sp_{unit_name}.npy",
            "prob_folder": f"./NN_model_NS_{NS}_ideal_prob_2",
            "label": "NN-joint(ideal)",
            "color": "green"
        },
        # "s_NN_500": {
        #     "fm1_path": f"./NN_model_NS_{NS}_FM1_ideal_500_2/survivals_probs_loss{NN_loss_type}_t{t}_ndata500/sp_{unit_name}.npy",
        #     "fm2_path": f"./NN_model_NS_{NS}_FM2_ideal_500_2/survivals_probs_loss{NN_loss_type}_t{t}_ndata500/sp_{unit_name}.npy",
        #     "prob_folder": f"./NN_model_NS_{NS}_ideal_500_prob_2",
        #     "label": "NN-joint(ideal)-500",
        #     "color": "purple"
        # },
        "s_NN_mis": {
            "fm1_path": f"./NN_model_NS_{NS}_FM1_misspec_2/survivals_probs_loss{NN_loss_type}_t{t}_ndata50/sp_{unit_name}.npy",
            "fm2_path": f"./NN_model_NS_{NS}_FM2_misspec_2/survivals_probs_loss{NN_loss_type}_t{t}_ndata50/sp_{unit_name}.npy",
            "prob_folder": f"./NN_model_NS_{NS}_misspec_prob_2",
            "label": "NN-joint(mis)",
            "color": "red"
        },
        # "s_NN_500_mis": {
        #     "fm1_path": f"./NN_model_NS_{NS}_FM1_misspec_500_2/survivals_probs_loss{NN_loss_type}_t{t}_ndata500/sp_{unit_name}.npy",
        #     "fm2_path": f"./NN_model_NS_{NS}_FM2_misspec_500_2/survivals_probs_loss{NN_loss_type}_t{t}_ndata500/sp_{unit_name}.npy",
        #     "prob_folder": f"./NN_model_NS_{NS}_misspec_500_prob_2",
        #     "label": "NN-joint(mis)-500",
        #     "color": "green"
        # },
    }

    for key, dataset in datasets.items():
        s_fm1 = np.load(dataset["fm1_path"])
        s_fm2 = np.load(dataset["fm2_path"])
        max_cols = max(s_fm1.shape[1], s_fm2.shape[1])
        if s_fm1.shape[1] < max_cols:
            s_fm1 = np.hstack((s_fm1, np.zeros((s_fm1.shape[0], max_cols - s_fm1.shape[1]))))
        if s_fm2.shape[1] < max_cols:
            s_fm2 = np.hstack((s_fm2, np.zeros((s_fm2.shape[0], max_cols - s_fm2.shape[1]))))
        probabilities_df = pd.read_csv(
            f'{dataset["prob_folder"]}/fm_probabilities_alpha10000_2000epoch/t_{t}/unit_{unit_name}_probabilities.csv')
        p1 = probabilities_df['fm_1_prob'].values[0]
        p2 = probabilities_df['fm_2_prob'].values[0]
        num_samples_fm1 = int(1000 * p1.sum())
        num_samples_fm2 = 1000 - num_samples_fm1
        selected_indices_fm1 = np.random.choice(len(s_fm1), size=num_samples_fm1, replace=False)
        selected_indices_fm2 = np.random.choice(len(s_fm2), size=num_samples_fm2, replace=False)
        s_combined = np.vstack((s_fm1[selected_indices_fm1], s_fm2[selected_indices_fm2]))

        est_sz_mean = np.median(s_combined, axis=0)
        est_sz_lower = np.quantile(s_combined, 0.025, axis=0)
        est_sz_upper = np.quantile(s_combined, 0.975, axis=0)

        mean_bounds_dict[key] = {
            'mean': est_sz_mean,
            'lower': est_sz_lower,
            'upper': est_sz_upper
        }

    df = pd.read_csv(file_path, delimiter='\t')

    for key, bounds in mean_bounds_dict.items():
        label = datasets[key]["label"].replace('-', '_').replace(' ', '_')
        df[f'est_sz_mean_{label}'] = bounds['mean'][:len(df)]
        df[f'est_sz_lower_{label}'] = bounds['lower'][:len(df)]
        df[f'est_sz_upper_{label}'] = bounds['upper'][:len(df)]

    df.to_csv(file_path, sep='\t', index=False)

    # plt.figure(figsize=(10, 6))
    # plt.plot(range(t_star, t_max), survival_means, marker='o', label='CMGP-Cox')
    # plt.fill_between(range(t_star, t_max), survival_lower_bounds, survival_upper_bounds, color='b', alpha=0.2,
    #                  label='CMGP-Cox - 95% CI')
    #
    # plt.plot(range(t_star, t_max), survival, marker='o', linestyle='--', label='True Survival Probability',
    #          color='darkorange')
    #
    # for key, bounds in mean_bounds_dict.items():
    #     label = datasets[key]["label"]
    #     plt.plot(range(t_star, t_max), bounds['mean'], label=label, color=datasets[key]["color"])
    #     plt.fill_between(range(t_star, t_max), bounds['lower'], bounds['upper'], alpha=0.2, color=datasets[key]["color"])
    #
    # plt.xlabel('Time')
    # plt.ylabel('Probability of Survival')
    # plt.title(f'Survival Probability with 95% Confidence Interval - unit ID:{unit_name}')
    # plt.grid(True)
    # plt.legend()
    # survival_plot_path = os.path.join(survival_plot_dir, f'survival_plot_unit_{unit_name}.png')
    # plt.savefig(survival_plot_path)
    # plt.close()
    #
    # plt.boxplot(abs_errors_for_unit)
    # plt.title(f'Absolute Error i:{unit_name}_{unit_mae[unit_name]} ')
    # plt.xlabel('Unit')
    # plt.ylabel('Absolute Error')
    # plt.grid(True)
    # box_plot_path = os.path.join(box_plot_dir, f'boxplot_{unit_name}.png')
    # plt.savefig(box_plot_path)
    # plt.close()


########################################################################################################################
import time

end_time = time.time()
execution_time = end_time - start_time  # Calculate the elapsed time in seconds

# Log the execution time
log_message(f"Total Execution Time: {execution_time:.5f} seconds")
