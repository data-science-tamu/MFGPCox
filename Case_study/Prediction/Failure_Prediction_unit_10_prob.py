
from utils.utils_final_fm.plot_save_print_fm import *
from utils.utils_final_fm.data_processing_fm import *
from utils.utils_final_fm.rul_pred_fm import *
from utils.utils_final_fm.options_fm import *
import logging
import time


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

########################################################################################################################
base_dir = os.path.dirname(os.path.abspath(__file__))
########################################################################################################################

t = 10

all_data_transformed = pd.read_csv(os.path.join(base_dir, f"historical_plus_test_data_{t}.csv"))

train_data_transformed = pd.read_csv(os.path.join(base_dir, "historical_data.csv"))

all_data_transformed['time, in cycles'] = all_data_transformed['time, in cycles'].astype('float32')
train_data_transformed['time, in cycles'] = train_data_transformed['time, in cycles'].astype('float32')

sensors_list = ["NL", "NH", "P13", "P26", "P3", "T3", "T6", "T42"]

sensors_str = "_".join(sensors_list)

save_directory = f'{sensors_str}_t_{t}_cprob_minV'

# if os.path.exists(save_directory):
#     raise FileExistsError(f"The directory '{save_directory}' already exists.")
# else:
os.makedirs(save_directory, exist_ok=True)
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

failure_modes_sensors = [(sensor, fm) for sensor in sensors for fm in failure_modes]

(train_units_event_time, test_units_event_time, min_V_by_failure_mode, unit_status, unit_manufacturer,
 unit_failure_mode) = process_unit_data(train_data_transformed, all_data_transformed, device)
########################################################################################################################
test_data_transformed = pd.read_csv(os.path.join(base_dir, f"test_data_{t}.csv"))
test_data_transformed['time, in cycles'] = test_data_transformed['time, in cycles'].astype('float32')

data_dicts_tests, _, _ = create_data_dicts(test_data_transformed, sensors_list)

(_, _, _, _, _,
 unit_failure_mode) = process_unit_data(test_data_transformed, test_data_transformed, device)
########################################################################################################################
# loading gp hyperparameters

loaded_hyperparameters = {}
loaded_lambda_hyp = {}

sensor_gp_hyperparameter_paths = {
    'NL': r"./units_NL_3f/optimized_hyperparameters_fm_specific_NL_2024-08-25_01"
          r"-34-27/iteration_25000/optimized_hyperparameters_fm_specific_NL_25000.pth",

    'NH': r"./units_NH_3f/optimized_hyperparameters_fm_specific_NH_2024-08-25_01"
          r"-33-27/iteration_25000/optimized_hyperparameters_fm_specific_NH_25000.pth",

    'P13': r"./units_P13_3f/optimized_hyperparameters_fm_specific_2024-08-09_19"
           r"-00-57/iteration_25000/optimized_hyperparameters_fm_specific_25000.pth",

    'P26': r"./units_P26_3f/optimized_hyperparameters_fm_specific_2024-09-03_14"
           r"-47-19/iteration_25000/optimized_hyperparameters_fm_specific_25000.pth",

    'P3': r"./units_P3_3f/optimized_hyperparameters_fm_specific_P3_2024-08-25_01"
          r"-34-53/iteration_25000/optimized_hyperparameters_fm_specific_P3_25000.pth",

    'T3': r"./units_T3_3f/optimized_hyperparameters_fm_specific_T3_2024-08-25_01"
          r"-35-59/iteration_25000/optimized_hyperparameters_fm_specific_T3_25000.pth",

    'T6': r"./units_T6_3f/optimized_hyperparameters_fm_specific_T6_2024-08-25_01"
          r"-36-22/iteration_25000/optimized_hyperparameters_fm_specific_T6_25000.pth",

    'T42': r"./units_T42_3f/optimized_hyperparameters_fm_specific_2024-08-09_22"
           r"-10-05/iteration_25000/optimized_hyperparameters_fm_specific_25000.pth"
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

loaded_params = torch.load(os.path.join(base_dir, r"optimized_parameters_2025-01-20_21-52-45"
                                                  r"/iteration_100000/optimized_parameters_Iter_100000_minus_min_v_100000.pth"),
                           map_location=device)
# loaded_params = torch.load(os.path.join(base_dir, r"optimized_parameters_2025"
#                                                   r"-01-19_17-19-48/iteration_50000/optimized_parameters_50000.pth"),
#                            map_location=device)


loaded_pi_hat = loaded_params['optimized_pi_hat']
loaded_mu_b_hat = loaded_params['optimized_mu_b_hat']
loaded_sigma_b_hat = loaded_params['optimized_sigma_b_hat']
loaded_alpha_rho_hat = loaded_params['optimized_alpha_rho_hat']
loaded_beta_rho_hat = loaded_params['optimized_beta_rho_hat']
loaded_alpha_hat = loaded_params['optimized_alpha_hat']
loaded_gamma = loaded_params['optimized_gamma']
loaded_beta = loaded_params['optimized_beta']

# loaded_pi_hat[1] = torch.tensor(0.5, device=device)
# loaded_pi_hat[2] = torch.tensor(0.5, device=device)

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


unit_f = test_data_transformed[['unit number', 'failure mode']].drop_duplicates().sort_values(by=['unit number'])
unit_f = list(unit_f.itertuples(index=False, name=None))

start_time = time.time()

unit_f_probabilities = {}

pi1 = loaded_pi_hat.get(1)
pi2 = loaded_pi_hat.get(2)
pi3 = loaded_pi_hat.get(3)

df_results = pd.DataFrame(columns=['Unit', 'P1', 'P2', 'P3'])

for unit in unit_f:
    unit_probs = []

    # Loop through each sensor for the current unit
    for sensor in sensors_list:
        # Call your function to get probabilities for each failure mode
        prob1, prob2, prob3 = likelihood_of_failure_all_test(data_dicts_tests, data_dicts, all_sensor_readings,
                                                             loaded_hyperparameters, loaded_lambda_hyp,
                                                             approx_cov_results, unit[0], unit[1], loaded_pi_hat,
                                                             sensor)

        unit_probs.append((prob1, prob2, prob3))

        constant = 1/1e5
    # Combine probabilities for each failure mode across all sensors for this unit
    combined_prob1 = torch.prod(torch.tensor([prob1 * constant for prob1, _, _ in unit_probs]))
    combined_prob2 = torch.prod(torch.tensor([prob2 * constant for _, prob2, _ in unit_probs]))
    combined_prob3 = torch.prod(torch.tensor([prob3 * constant for _, _, prob3 in unit_probs]))

    # Sum of combined probabilities for normalization

    total_prob = pi1 * combined_prob1 + pi2 * combined_prob2 + pi3 * combined_prob3

    # Normalize to get final probabilities for each failure mode
    final_prob1 = combined_prob1 * pi1 / total_prob
    final_prob2 = combined_prob2 * pi2 / total_prob
    final_prob3 = combined_prob3 * pi3 / total_prob

    # Store the results
    unit_f_probabilities[unit[0]] = {1: final_prob1, 2: final_prob2, 3: final_prob3}
    log_message(
        f'Unit:{unit[0]} ===> P1: {final_prob1.item()},  P2: {final_prob2.item()},  P3: {final_prob3.item()}')

    df_results = df_results._append({'Unit': int(unit[0]),
                                     'P1': final_prob1.item(),
                                     'P2': final_prob2.item(),
                                     'P3': final_prob3.item()},
                                    ignore_index=True)
log_message('\n')

for unit, probabilities in unit_f_probabilities.items():
    failure_mode = max(probabilities, key=probabilities.get)

    log_message(f"Unit: {unit} ==> Failure Mode: {failure_mode}")

    df_results.loc[df_results['Unit'] == unit, 'Predicted_Failure_Mode'] = failure_mode

actual_failure_modes = {
    **{unit: 1 for unit in range(11, 31)},
    **{unit: 2 for unit in range(41, 101)},
    **{unit: 3 for unit in range(111, 151)},
}

for unit, value in actual_failure_modes.items():
    df_results.loc[df_results['Unit'] == unit, 'Actual_Failure_Mode'] = value

log_message('\n')

df_results.to_csv(os.path.join(save_directory, f'failure_mode_probs_{t}.csv'), index=False)

log_message('Results saved to failure_mode_probs.csv')

predicted_failure_modes = {}
correct_predictions = 0
total_predictions = 0

correct_per_mode = {1: 0, 2: 0, 3: 0}
total_per_mode = {1: 0, 2: 0, 3: 0}

pred_fm = []

for unit, probabilities in unit_f_probabilities.items():

    predicted_failure_mode = max(probabilities, key=probabilities.get)
    pred_fm.append(predicted_failure_mode)

    actual_failure_mode = actual_failure_modes.get(unit)

    total_predictions += 1
    if predicted_failure_mode == actual_failure_mode:
        correct_predictions += 1

    if actual_failure_mode in correct_per_mode:
        total_per_mode[actual_failure_mode] += 1
        if predicted_failure_mode == actual_failure_mode:
            correct_per_mode[actual_failure_mode] += 1

overall_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

accuracy_per_mode = {mode: correct_per_mode[mode] / total_per_mode[mode] if total_per_mode[mode] > 0 else 0
                     for mode in correct_per_mode}

log_message(f"Overall Accuracy: {overall_accuracy:.2f}")

log_message('\n')

for mode, accuracy in accuracy_per_mode.items():
    log_message(f"Accuracy for Failure Mode {mode}: {accuracy:.2f}")

# unit_f_probabilities = {
#     **{i: {1: torch.tensor(1.0), 2: torch.tensor(0.0), 3: torch.tensor(0.0)} for i in range(1, 31)},
#     **{i: {1: torch.tensor(0.0), 2: torch.tensor(1.0), 3: torch.tensor(0.0)} for i in range(31, 101)},
#     **{i: {1: torch.tensor(0.0), 2: torch.tensor(0.0), 3: torch.tensor(1.0)} for i in range(101, 151)},
# }
#
# predicted_failure_modes = actual_failure_modes
########################################################################################################################
from utils.utils_final.plot_save_print import *
from utils.utils_final.data_processing import *
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

        'NH': r"./units_NH_3f/optim_test_hyperparams_10_all_2025-01-22_01-39-50"
              r"/iteration_25000/optim_test_hyperparams_10_all_25000.pth",

        'NL': r"./units_NL_3f/optim_test_hyperparams_10_all_2025-01-22_01-41-04"
              r"/iteration_25000/optim_test_hyperparams_10_all_25000.pth",

        'P3': r"./units_P3_3f/optim_test_hyperparams_10_all_2025-01-22_01-42-02"
              r"/iteration_25000/optim_test_hyperparams_10_all_25000.pth",

        'P13': r"./units_P13_3f/optim_test_hyperparams_10_all_2025-01-22_01-45"
               r"-16/iteration_25000/optim_test_hyperparams_10_all_25000.pth",

        'P26': r"./units_P26_3f/optim_test_hyperparams_10_all_2025-01-22_01-46"
               r"-17/iteration_25000/optim_test_hyperparams_10_all_25000.pth",

        'T3': r"./units_T3_3f/optim_test_hyperparams_10_all_2025-01-22_01-48-06"
              r"/iteration_25000/optim_test_hyperparams_10_all_25000.pth",

        'T6': r"./units_T6_3f/optim_test_hyperparams_10_all_2025-01-22_22-33-14"
              r"/iteration_25000/optim_test_hyperparams_10_all_25000.pth",

        'T42': r"./units_T42_3f/optim_test_hyperparams_10_all_2025-01-22_22-33"
               r"-47/iteration_25000/optim_test_hyperparams_10_all_25000.pth"
    }

if t == 25:
    sensor_gp_hyperparameter_paths = {

        'NH': r"./units_NH_3f/optim_test_hyperparams_25_2025-01-19_14-02-08"
              r"/iteration_25000/optim_test_hyperparams_25_25000.pth",

        'NL': r"./units_NL_3f/optim_test_hyperparams_25_2025-01-19_16-21-27"
              r"/iteration_10000/optim_test_hyperparams_25_10000.pth",

        'P3': r"./units_P3_3f/optim_test_hyperparams_25_2025-01-19_16-22-55"
              r"/iteration_25000/optim_test_hyperparams_25_25000.pth",

        'P13': r"./units_P13_3f/optim_test_hyperparams_25_2025-01-19_16-23-19/iteration_25000"
               r"/optim_test_hyperparams_25_25000.pth",

        'P26': r"./units_P26_3f/optim_test_hyperparams_25_2025-01-19_23-37-28/iteration_25000"
               r"/optim_test_hyperparams_25_25000.pth",

        'T3': r"./units_T3_3f/optim_test_hyperparams_25_2025-01-19_23-42-23/iteration_25000"
              r"/optim_test_hyperparams_25_25000.pth",

        'T6': r"./units_T6_3f/optim_test_hyperparams_25_2025-01-20_09-42-30/iteration_20000"
              r"/optim_test_hyperparams_25_20000.pth",

        'T42': r"./units_T42_3f/optim_test_hyperparams_25_2025-01-20_09-44-45/iteration_25000"
               r"/optim_test_hyperparams_25_25000.pth"
    }

for sensor, path in sensor_gp_hyperparameter_paths.items():
    loaded_hyperparams = torch.load(path, map_location=device)
    loaded_hyperparameters.update(loaded_hyperparams['optimized_hyperparameters'])
    optimized_lambda_hyp = loaded_hyperparams.get('optimized_lambda_hyp', )

    for i in range(1, 4):
        if i in optimized_lambda_hyp:
            loaded_lambda_hyp[(sensor, i)] = optimized_lambda_hyp[i]

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

test_data_all = pd.read_csv(os.path.join(base_dir, "test_data_all.csv"))

current_time = t

failure_times = test_data_all.groupby('unit number')['time, in cycles'].max().reset_index()
failure_times.rename(columns={'time, in cycles': 'failure time'}, inplace=True)

unit_t = [(int(row['unit number']), current_time, row['failure time'] - current_time) for _, row in
          failure_times.iterrows()]

log_message(unit_t)
########################################################################################################################


number_of_samples = 10000
log_message(f'number of samples: {number_of_samples}')
plots_directory = os.path.join(save_directory, f"plots_{t}_exact_ns{number_of_samples}")
survivals_directory = os.path.join(save_directory, f"sp_{t}_exact_ns{number_of_samples}")

survival_plot_dir = os.path.join(plots_directory, 'survival')
box_plot_dir = os.path.join(plots_directory, 'boxplot')

os.makedirs(survival_plot_dir, exist_ok=True)
os.makedirs(box_plot_dir, exist_ok=True)

os.makedirs(survivals_directory, exist_ok=True)
os.makedirs(plots_directory, exist_ok=True)

absolute_errors = []
unit_mae = {}
for unit in unit_t:
    t_star = unit[1]
    unit_name = unit[0]
    if unit_name in range(111, 151):
        continue
    print(unit_name)
    actual_failure = unit[1] + unit[2]
    failure_mode = actual_failure_modes.get(unit_name)
    t_max = 220

    # survival = []
    failure = []
    predicted_survival = []
    # abs_errors_for_unit = []
    survival_means = []
    survival_lower_bounds = []
    survival_upper_bounds = []

    file_path = os.path.join(survivals_directory, f'{unit_name}.tsv')
    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerow(
            ['Time', 'Failure_Probability', 'Survival_Probability', 'lower_bound', 'Upper_bound', 'Actual_Failure'])

        for time in range(t_star, t_max):
            print(time)
            survival_distribution = St_cond_EST_prob_batch_cs(min_V_by_failure_mode, unit_f_probabilities,
                                                              failure_modes,
                                                              all_sensor_readings, approx_cov_results,
                                                              unit_manufacturer,
                                                              sensors, unit_name, t_star, time, loaded_mu_b_hat,
                                                              loaded_sigma_b_hat, loaded_alpha_rho_hat,
                                                              loaded_beta_rho_hat, loaded_beta, loaded_gamma,
                                                              data_dicts, loaded_hyperparameters, loaded_lambda_hyp,
                                                              number_of_samples)

            mean_survival = torch.mean(survival_distribution).item()
            lower_bound = torch.quantile(survival_distribution, 0.025).item()
            upper_bound = torch.quantile(survival_distribution, 0.975).item()
            survival_means.append(mean_survival)
            survival_lower_bounds.append(lower_bound)
            survival_upper_bounds.append(upper_bound)

            failure_probability = 1 - mean_survival
            predicted_survival.append(mean_survival)
            failure.append(failure_probability)

            writer.writerow(
                [time, failure_probability, mean_survival, lower_bound, upper_bound, actual_failure])

    # plt.figure(figsize=(10, 6))
    # plt.plot(range(t_star, t_max), survival_means, marker='o', label='CMGP-Cox')
    # plt.fill_between(range(t_star, t_max), survival_lower_bounds, survival_upper_bounds, color='b', alpha=0.2,
    #                  label='CMGP-Cox - 95% CI')
    # plt.xlabel('Time')
    # plt.ylabel('Probability of Survival')
    # plt.title(f'Survival Probability with 95% Confidence Interval - unit ID:{unit_name}')
    # plt.grid(True)
    # plt.legend()
    # survival_plot_path = os.path.join(survival_plot_dir, f'survival_plot_unit_{unit_name}.png')
    # plt.savefig(survival_plot_path)
    # plt.close()

print()
########################################################################################################################

import time

end_time = time.time()
execution_time = end_time - start_time

log_message(f"Total Execution Time: {execution_time:.5f} seconds")
