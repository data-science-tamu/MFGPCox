from torch.special import digamma
import torch.optim as optim
from datetime import datetime
from utils.utils_final.plot_save_print import *
from utils.utils_final.data_processing import *
from utils.utils_final.CMGP import *
from utils.utils_final.options import *
from torch import lgamma
import neptune

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

########################################################################################################################
base_dir = os.path.dirname(os.path.abspath(__file__))
########################################################################################################################
########################################################################################################################
inducing_points_num = 128

# assign random seed
seed_value = 423
torch.manual_seed(seed_value)

########################################################################################################################


# Loading Data

## transformed data
all_data_transformed = pd.read_csv(os.path.join(base_dir, "historical_data.csv"))
train_data_transformed = pd.read_csv(os.path.join(base_dir, "historical_data.csv"))

all_data_transformed['time, in cycles'] = all_data_transformed['time, in cycles'].astype('float32')
train_data_transformed['time, in cycles'] = train_data_transformed['time, in cycles'].astype('float32')

sensors_list = ["sensor 1", "sensor 2"]
data_dicts, all_sensor_readings, all_time_points = create_data_dicts(all_data_transformed, sensors_list)

failure_modes = all_data_transformed['failure mode'].unique()

sensors = all_data_transformed.columns.intersection(
    sensors_list)  # sensors_list contains the names of the sensors you are interested in

# Generate a list of (sensor, failure_mode) combinations
failure_modes_sensors = [(sensor, fm) for sensor in sensors for fm in failure_modes]

########################################################################################################################
# All observations

train_units_event_time = {}
test_units_event_time = {}

unique_historical_units = train_data_transformed['unit number'].unique()
unique_units = all_data_transformed['unit number'].unique()

train_unit_range = unique_historical_units

for unit in all_data_transformed['unit number'].unique():
    subset_df = all_data_transformed[all_data_transformed['unit number'] == unit]
    max_event_time = subset_df['time, in cycles'].max()
    event_time_tensor = torch.tensor(max_event_time, dtype=torch.float32, device=device)
    if unit in train_unit_range:
        train_units_event_time[unit] = event_time_tensor
    else:
        test_units_event_time[unit] = event_time_tensor

# V = torch.tensor(list(train_units_event_time.values()))
# min_V = V.min()

failure_mode_groups = train_data_transformed.groupby('failure mode')

min_V_by_failure_mode = {}
for failure_mode, group in failure_mode_groups:
    train_units_event_time_mode = {
        unit: train_units_event_time[unit]
        for unit in group['unit number'].unique()
        if unit in train_units_event_time
    }
    if train_units_event_time_mode:
        V = torch.tensor(list(train_units_event_time_mode.values()))
        min_V_by_failure_mode[failure_mode] = V.min()

# print()

# Create a dictionary with unit as the key and status(event type(failure = 1 , Censored = 0)) value

unit_status = {}
for unit in all_data_transformed['unit number'].unique():
    subset_df = all_data_transformed[all_data_transformed['unit number'] == unit]
    # max_event_time = subset_df['time, in cycles'].max()
    unit_status[unit] = torch.tensor(1, dtype=torch.float32, device=device)

all_units = unique_units.tolist()
historical_units = unique_historical_units.tolist()
test_units = [unit for unit in all_units if unit not in historical_units]

unit_manufacturer = {}
for unit in all_data_transformed['unit number'].unique():
    subset_df = all_data_transformed[all_data_transformed['unit number'] == unit]
    # max_event_time = subset_df['time, in cycles'].max()
    unit_manufacturer[unit] = torch.tensor(0, dtype=torch.float32, device=device)

unit_failure_mode = {}
for unit in all_data_transformed['unit number'].unique():
    subset_df = all_data_transformed[all_data_transformed['unit number'] == unit]
    failure = subset_df['failure mode'].max()
    unit_failure_mode[unit] = failure

########################################################################################################################

sensor_gp_hyperparameter_paths = {
    'sensor 1': r"sensor_1/optimized_hyperparameters_unit_specific_2025-01-03_11-47-46/iteration_10000"
                r"/optimized_hyperparameters_unit_specific_xi_10000.pth",

    'sensor 2': r"sensor_2/optim_hyperparam_unit_historical_res2_2025-01-01_12-24-20/iteration_200000"
                r"/optimized_hyperparameters_unit_specific_historical_res2_200000.pth"
}

# sensor_gp_hyperparameter_paths = {
#     'sensor 1': r"sensor_1/optim_test_hyperparams_20_final_2025-01-03_14-09-20/iteration_45000/optim_test_hyperparams_20_final_45000.pth",
#
#     'sensor 2': r"sensor_2/optim_test_hyperparams_20_final_2025-01-03_14-19-20/iteration_45000/optim_test_hyperparams_20_final_45000.pth"
# }


loaded_hyperparameters = {}
loaded_lambda_hyp = {}

for sensor, path in sensor_gp_hyperparameter_paths.items():
    loaded_hyperparams = torch.load(os.path.join(base_dir, path), map_location=device, weights_only=False)

    loaded_hyperparameters.update(loaded_hyperparams['optimized_hyperparameters'])

    optimized_lambda_hyp = loaded_hyperparams.get('optimized_lambda_hyp', {})

    for i in range(1, 4):
        if i in optimized_lambda_hyp:
            loaded_lambda_hyp[(sensor, i)] = optimized_lambda_hyp[i]

########################################################################################################################


# gamma
random_values = torch.rand(len(failure_modes), device=device)
initial_gamma = {mode: value for mode, value in zip(failure_modes, random_values)}

# alpha_hat
random_values = torch.rand(len(failure_modes), device=device)
initial_alpha_hat = {mode: value for mode, value in zip(failure_modes, random_values)}

random_values = torch.rand(len(failure_modes_sensors), device=device)
initial_beta = {(sensor, fm): value for (sensor, fm), value in zip(failure_modes_sensors, random_values)}

random_values = torch.rand(len(failure_modes), device=device)
alpha0 = {mode: value for mode, value in zip(failure_modes, random_values)}
alpha0[1] = torch.tensor(0.1, device=device)
alpha0[2] = torch.tensor(0.1, device=device)

initial_beta[('sensor 1', 1)] = torch.tensor(0.01, device=device)
initial_beta[('sensor 1', 2)] = torch.tensor(0.01, device=device)
initial_beta[('sensor 2', 1)] = torch.tensor(0.01, device=device)
initial_beta[('sensor 2', 2)] = torch.tensor(0.01, device=device)

mu_b_0 = {1: torch.tensor(-7.0, device=device), 2: torch.tensor(-10.0, device=device)}
sigma_b_0 = {1: torch.tensor(0.5, device=device), 2: torch.tensor(0.5, device=device)}
alpha_rho_0 = {1: torch.tensor(0.015 * 300, device=device), 2: torch.tensor(0.0095 * 300, device=device)}
beta_rho_0 = {1: torch.tensor(1.0 * 300, device=device), 2: torch.tensor(1.0 * 300, device=device)}

random_values = torch.rand(len(failure_modes), device=device)
initial_mu_b_hat = {mode: value for mode, value in zip(failure_modes, random_values)}
initial_mu_b_hat[1] = torch.tensor(-6, device=device)
initial_mu_b_hat[2] = torch.tensor(-8, device=device)

random_values = torch.rand(len(failure_modes), device=device)
initial_sigma_b_hat = {mode: value for mode, value in zip(failure_modes, random_values)}
initial_sigma_b_hat[1] = torch.tensor(0.16, device=device)
initial_sigma_b_hat[2] = torch.tensor(0.16, device=device)

random_values = torch.rand(len(failure_modes), device=device)
initial_alpha_rho_hat = {mode: value for mode, value in zip(failure_modes, random_values)}

initial_alpha_rho_hat[1] = torch.log(torch.tensor(0.022 * 300, device=device))
initial_alpha_rho_hat[2] = torch.log(torch.tensor(0.07 * 300, device=device))
#

random_values = torch.rand(len(failure_modes), device=device)
initial_beta_rho_hat = {mode: value for mode, value in zip(failure_modes, random_values)}

initial_beta_rho_hat[1] = torch.log(torch.tensor(1 * 300, device=device))
initial_beta_rho_hat[2] = torch.log(torch.tensor(1 * 300, device=device))

########################################################################################################################


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
# ELBO: Cox Model terms


def h0(l, mu_b_hat, sigma_b_hat, alpha_rho_hat, beta_rho_hat, min_V):
    return torch.exp((mu_b_hat + ((sigma_b_hat ** 2) / 2))) * (1 - (l / beta_rho_hat)) ** (
        -alpha_rho_hat)


# def h0(l, mu_b_hat, sigma_b_hat, alpha_rho_hat, beta_rho_hat, min_V):
#     return torch.where(l >= min_V,
#                        torch.exp((mu_b_hat + ((sigma_b_hat ** 2) / 2))) * (1-((l - min_V)/beta_rho_hat))**(-alpha_rho_hat),
#                        torch.tensor(0.0))


precomputed_means = {}
for unit in historical_units:
    unit_failure = unit_failure_mode.get(unit)
    unit_event_time = train_units_event_time.get(unit).unsqueeze(0)
    mean, _ = get_cmgp_predictions(
        all_sensor_readings, unit_event_time, inducing_points_num, unit,
        sensors_list, unit_failure, data_dicts, loaded_hyperparameters,
        loaded_lambda_hyp, approx_cov_results, preferred_device=device
    )
    precomputed_means[(unit, unit_failure)] = mean.squeeze()


def get_cox1(data, sensor, mu_b_dict, alpha_rho_dict, beta_rho_dict, beta_dict, gamma_dict,
             hyperparameters,
             lambda_hyperparameter):
    cox1 = torch.tensor(0.0, device=device).unsqueeze(-1)

    for unit in historical_units:
        unit_failure = unit_failure_mode.get(unit)
        mu_b = mu_b_dict.get(unit_failure)
        alpha_rho = alpha_rho_dict.get(unit_failure)
        beta_rho = beta_rho_dict.get(unit_failure)

        beta_values = []
        for sensor in sensors:
            beta_value = beta_dict.get((sensor, unit_failure))
            if beta_value is not None:
                beta_values.append(beta_value)
            else:
                raise ValueError(f"Beta value for sensor {sensor} and failure mode {unit_failure} not found.")

        beta = torch.stack(beta_values).to(device)
        gamma = gamma_dict.get(unit_failure)
        # pi_hat = pi_hat_dict.get(unit_failure)
        # min_V = min_V_by_failure_mode.get(unit_failure)

        status = unit_status.get(unit)
        manufacturer = unit_manufacturer.get(unit)
        unit_event_time = train_units_event_time.get(unit)
        # first = (mu_b + (alpha_rho / beta_rho) * (
        #         unit_event_time - min_V)) if unit_event_time >= min_V else torch.tensor(0.0)

        first = (mu_b + (alpha_rho / beta_rho) * unit_event_time)
        # second = gamma * manufacturer
        # print(unit, min_V)
        # print(unit_event_time - min_V)

        # Retrieve the precomputed mean for the unit and failure mode
        mean = precomputed_means[(unit, unit_failure)]
        # print(mean)

        # Ensure beta and mean have the same dimensions
        if beta.shape != mean.shape:
            raise ValueError(f"Shape mismatch between beta {beta.shape} and mean {mean.shape} for unit {unit}.")

        third = beta @ mean
        cox1 += status * (first + third)

    return cox1


precomputed_predictions = {}


def precompute_predictions(units, sensors, data, hyperparameters, lambda_hyperparameter):
    """
    Pre-computes mean and variance predictions for each unit given their event times.
    Stores results in the precomputed_predictions dictionary.
    """
    for unit in units:
        unit_failure = unit_failure_mode.get(unit)
        unit_event_time = (train_units_event_time if unit in historical_units else test_units_event_time).get(unit)
        num_points = 1000
        ls = torch.linspace(0, unit_event_time, num_points).to(device)

        # Compute means and variances once for all l values
        means, variances = get_cmgp_predictions(
            all_sensor_readings, ls, inducing_points_num, unit, sensors, unit_failure,
            data, hyperparameters, lambda_hyperparameter, approx_cov_results, preferred_device=device
        )

        # print()
        # Store results in the dictionary using unit as key
        precomputed_predictions[unit] = {
            'ls': ls,
            'means': means.squeeze(),
            'variances_diagonal': torch.diagonal(variances, dim1=1, dim2=2).squeeze()
        }


def integrand(precomputed, mu_b_hat, sigma_b_hat, alpha_rho_hat, beta_rho_hat, beta, gamma, unit):
    """
    Modified integrand function to use precomputed mean and variance values.
    """
    h0_value = h0(precomputed['ls'], mu_b_hat, sigma_b_hat, alpha_rho_hat, beta_rho_hat,
                  min_V_by_failure_mode.get(unit_failure_mode.get(unit)))
    # if unit == 1:
    #     print(beta)
    exp_component_first = beta @ precomputed['means']

    exp_component_second = 0.5 * beta ** 2 @ precomputed['variances_diagonal']
    # exp_component_third = gamma * unit_manufacturer.get(unit)
    exp_component = exp_component_first + exp_component_second  #+ exp_component_third

    return h0_value * torch.exp(exp_component)


def get_cox2(sensors, mu_b_dict, sigma_b_dict, alpha_rho_dict, beta_rho_dict, beta_dict, gamma_dict,
             precomputed_predictions):
    cox2 = torch.tensor(0.0, device=device).unsqueeze(-1)

    # Process each unit using precomputed predictions
    for unit in historical_units + test_units:
        unit_failure = unit_failure_mode.get(unit)

        mu_b = mu_b_dict.get(unit_failure)
        sigma_b = sigma_b_dict.get(unit_failure)
        alpha_rho = alpha_rho_dict.get(unit_failure)
        beta_rho = beta_rho_dict.get(unit_failure)

        # Collect beta values for the current unit and failure mode
        beta_values = [beta_dict.get((sensor, unit_failure)) for sensor in sensors]
        beta = torch.stack(beta_values).to(device)

        gamma = gamma_dict.get(unit_failure)
        # pi_hat = pi_hat_dict.get(unit_failure)

        # Retrieve precomputed predictions
        precomputed = precomputed_predictions[unit]
        vals = integrand(precomputed, mu_b, sigma_b, alpha_rho, beta_rho, beta, gamma, unit)

        # Perform the integral approximation
        integral_approx = torch.trapz(vals, precomputed['ls'])
        cox2 += integral_approx
        # print(f'unit:{unit} ==> cox2:{cox2.item()}')
    return cox2


all_units = historical_units + test_units
precompute_predictions(all_units, sensors, data_dicts, loaded_hyperparameters, loaded_lambda_hyp)


########################################################################################################################


def kl_term(data, alpha_hat_dict):  # `data` kept in the signature

    alpha_hat_vector = torch.stack(list(alpha_hat_dict.values())).to(device)
    alpha0_vector = torch.stack(list(alpha0.values())).to(device)

    summation_first = torch.lgamma(torch.sum(alpha_hat_vector)) \
                      - torch.lgamma(torch.sum(alpha0_vector))

    summation_second = torch.sum(torch.lgamma(alpha0_vector) \
                                 - torch.lgamma(alpha_hat_vector))

    digamma_diff = torch.digamma(alpha_hat_vector) \
                   - torch.digamma(torch.sum(alpha_hat_vector))

    summation_third = torch.sum((alpha_hat_vector - alpha0_vector) * digamma_diff)

    negative_kl = -(summation_first + summation_second + summation_third)

    return negative_kl + torch.sum(digamma_diff)


def kl_b_rho(mu_b_0_dict, sigma_b_0_dict, alpha_rho_0_dict, beta_rho_0_dict, mu_b_hat_dict,
             sigma_b_hat_dict, alpha_rho_hat_dict, beta_rho_hat_dict):
    total_kl = torch.tensor(0.0, device=device).unsqueeze(-1)

    for failure_mode in failure_modes:
        mu_b = mu_b_0_dict[failure_mode]
        sigma_b = sigma_b_0_dict[failure_mode]
        alpha_rho = alpha_rho_0_dict[failure_mode]
        beta_rho = beta_rho_0_dict[failure_mode]
        mu_b_hat = mu_b_hat_dict[failure_mode]
        sigma_b_hat = sigma_b_hat_dict[failure_mode]
        alpha_rho_hat = alpha_rho_hat_dict[failure_mode]
        beta_rho_hat = beta_rho_hat_dict[failure_mode]

        kl_b = (1 / 2) * (torch.log((sigma_b / sigma_b_hat) ** 2) + (
                (sigma_b_hat ** 2 + (mu_b_hat - mu_b) ** 2) / (sigma_b ** 2)))

        kl_rho = ((alpha_rho_hat * torch.log(beta_rho_hat) - alpha_rho * torch.log(beta_rho))
                  - (lgamma(alpha_rho_hat) - lgamma(alpha_rho))
                  + (alpha_rho_hat - alpha_rho) * (digamma(alpha_rho_hat) - torch.log(beta_rho_hat))
                  - (beta_rho_hat - beta_rho) * (alpha_rho_hat / beta_rho_hat))

        total_kl += kl_b + kl_rho

    return total_kl


########################################################################################################################

def negative_elbo(sensor, mu_b_0_dict, sigma_b_0_dict, alpha_rho_0_dict, beta_rho_0_dict, mu_b_hat_dict,
                  sigma_b_hat_dict, alpha_rho_hat_dict, beta_rho_hat_dict, beta_dict, gamma_dict, alpha_dict,
                  data, hyperparameters,
                  lambda_hyp):
    cox_1 = get_cox1(data, sensor, mu_b_hat_dict, alpha_rho_hat_dict, beta_rho_hat_dict, beta_dict, gamma_dict,
                     hyperparameters, lambda_hyp)

    cox_2 = get_cox2(sensors_list, mu_b_hat_dict, sigma_b_hat_dict, alpha_rho_hat_dict, beta_rho_hat_dict, beta_dict,
                     gamma_dict, precomputed_predictions)

    kl = kl_term(data, alpha_dict)

    kl_of_b_and_rho = kl_b_rho(mu_b_0_dict, sigma_b_0_dict, alpha_rho_0_dict, beta_rho_0_dict, mu_b_hat_dict,
                               sigma_b_hat_dict, alpha_rho_hat_dict, beta_rho_hat_dict)

    return -(cox_1 - cox_2 + kl - kl_of_b_and_rho), -(cox_1 - cox_2), - kl, kl_of_b_and_rho


########################################################################################################################


########################################################################################################################
def objective_function(sensor, mu_b_0_dict, sigma_b_0_dict, alpha_rho_0_dict, beta_rho_0_dict, flat_parameters,
                       metadata, data, hyperparameters, lambda_hyp):
    reconstructed_params = reconstruct_hyperparameters_elbo_2(flat_parameters, metadata)
    reconstructed_mu_b_hat = reconstructed_params['mu_b_hat']
    reconstructed_sigma_b_hat = reconstructed_params['sigma_b_hat']
    reconstructed_alpha_rho_hat = reconstructed_params['alpha_rho_hat']
    reconstructed_beta_rho_hat = reconstructed_params['beta_rho_hat']
    reconstructed_alpha_hat = reconstructed_params['alpha_hat']
    reconstructed_beta = reconstructed_params['beta']
    reconstructed_gamma = reconstructed_params['gamma']

    neg_elbo, cox_loss, kl_loss, kl_b_rho_loss = negative_elbo(sensor, mu_b_0_dict, sigma_b_0_dict,
                                                               alpha_rho_0_dict, beta_rho_0_dict,
                                                               reconstructed_mu_b_hat,
                                                               reconstructed_sigma_b_hat,
                                                               reconstructed_alpha_rho_hat,
                                                               reconstructed_beta_rho_hat,
                                                               reconstructed_beta,
                                                               reconstructed_gamma,
                                                               reconstructed_alpha_hat, data,
                                                               hyperparameters, lambda_hyp)

    return neg_elbo, cox_loss, kl_loss, kl_b_rho_loss


flat_parameters_initial, metadata = flatten_hyperparameters_elbo_3(
    initial_mu_b_hat,
    initial_sigma_b_hat,
    initial_alpha_rho_hat,
    initial_beta_rho_hat,
    initial_alpha_hat,
    initial_beta,
    initial_gamma
)

flat_parameters_initial.requires_grad_(True)
# optimizer = optim.Adam([flat_parameters_initial], lr=0.01)
optimizer = optim.Adam([flat_parameters_initial], lr=0.01)
losses = []


########################################################################################################################


def closure():
    optimizer.zero_grad()
    total_loss, cox_loss, kl_loss, kl_b_rho_loss = objective_function(
        sensors_list, mu_b_0, sigma_b_0, alpha_rho_0, beta_rho_0,
        flat_parameters_initial, metadata, data_dicts, loaded_hyperparameters, loaded_lambda_hyp)


    total_loss.backward()
    losses.append((total_loss.item(), cox_loss.item(), kl_loss.item(), kl_b_rho_loss.item()))
    return total_loss


# Initialize timing variables
start_time = time.time()

num_iterations = 100000
header_written = False
now = datetime.now()
formatted_now = now.strftime("%Y-%m-%d_%H-%M-%S")
main_folder = base_dir
file_name = f"optimized_parameters_unit_2S_final_cottected_pi_m"
main_folder_path = os.path.join(main_folder, f"{file_name}_{formatted_now}")
# main_folder_path = os.path.join(main_folder, f"optimized_parameters_unit_2S_2024-12-01_13-12-12_Continue")
os.makedirs(main_folder_path, exist_ok=True)

for iteration in range(num_iterations):
    optimizer.step(closure)

    header_written = save_and_print_parameters_3(
        iteration=iteration,
        num_iterations=num_iterations,
        losses=losses,
        flat_parameters_initial=flat_parameters_initial,
        metadata=metadata,
        main_folder_path=main_folder_path,
        file_name=file_name,
        save_interval=1000,
        print_interval=100,
        header_written=header_written,
        start_time=start_time,
    )

