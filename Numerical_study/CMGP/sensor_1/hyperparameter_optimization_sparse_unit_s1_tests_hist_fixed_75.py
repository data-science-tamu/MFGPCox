import neptune
from neptune.utils import stringify_unsupported
import torch.optim as optim
from datetime import datetime
from utils.utils_distinct_gp_per_unit.plot_save_print_unit import *
from utils.utils_distinct_gp_per_unit.data_processing_unit import *
from utils.utils_distinct_gp_per_unit.CMGP_unit import *
from collections import defaultdict

########################################################################################################################


# seed_value = 42
# torch.manual_seed(seed_value)
inducing_points_num = 128


########################################################################################################################

# specifying the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

########################################################################################################################
base_dir = os.path.dirname(os.path.abspath(__file__))
########################################################################################################################
# Data
historical_test = pd.read_csv(os.path.join(base_dir, "historical_plus_test_data_updated_75.csv"))
test_data = pd.read_csv(os.path.join(base_dir, "test_data_75.csv"))

historical_test['time, in cycles'] = historical_test['time, in cycles'].astype('float32')
test_data['time, in cycles'] = test_data['time, in cycles'].astype('float32')

sensors_list = ['sensor 1']

data_dicts, all_signals, _ = create_data_dicts(historical_test, sensors_list)
data_dicts_test, _, _ = create_data_dicts(test_data, sensors_list)

failure_modes = historical_test['failure mode'].unique()

########################################################################################################################
# hyperparameters

loaded_hyperparams = torch.load(
    os.path.join(base_dir, r"optimized_hyperparameters_unit_specific_2025-01-03_11-47-46/iteration_10000"
                           r"/optimized_hyperparameters_unit_specific_xi_10000.pth"), map_location=device)

# Accessing the loaded hyperparameters
loaded_hyperparameters = loaded_hyperparams['optimized_hyperparameters']
loaded_lambda_hyp = loaded_hyperparams['optimized_lambda_hyp']

grouped_data = defaultdict(list)
for key, values in loaded_hyperparameters.items():
    group_key = key[1:]
    grouped_data[group_key].append(values)

loaded_hyperparameters_average = {}
for group_key, group_values in grouped_data.items():
    avg_values = {}
    for param in group_values[0].keys():  # Assume all entries in group have the same keys
        avg_values[param] = torch.mean(torch.stack([v[param] for v in group_values]), dim=0)
    loaded_hyperparameters_average[group_key] = avg_values



initial_hyperparameters_test, _ = generate_initial_hyperparameters_3(data_dicts_test)
test_keys = initial_hyperparameters_test.keys()

for test_key in test_keys:
    sensor = test_key[1]
    mode = test_key[2]
    average_key = (sensor, mode)
    if average_key in loaded_hyperparameters_average:
        initial_hyperparameters_test[test_key] = loaded_hyperparameters_average[average_key]
    else:
        print(f"Warning: No match found for test key {test_key} in loaded_hyperparameters_average")

# print()


########################################################################################################################


def objective_function(flat_parameters, number_of_inducing_points, lambda_hyp, hist_hyp, test_keys):
    hyperparameters = reconstruct_hyperparameters_test(flat_parameters, hist_hyp, test_keys)

    nll = {}
    for failure_mode, data_dict in data_dicts.items():
        # Retrieve the corresponding lambda_hyp for the current failure mode
        current_lambda_hyp = lambda_hyp[failure_mode]

        all_sensor_readings = all_signals.get(failure_mode)

        nll_fm = neg_log_likelihood(all_sensor_readings, data_dict, hyperparameters, current_lambda_hyp,
                                    number_of_inducing_points, device)
        nll[failure_mode] = nll_fm

    total_nll = sum(nll.values())

    individual_nlls = [nll[fm] for fm in failure_modes]

    return total_nll, *individual_nlls


########################################################################################################################




flat_parameters_initial = flatten_hyperparameters_test(initial_hyperparameters_test)

# hyperparameters = reconstruct_hyperparameters_test(flat_parameters_initial, loaded_hyperparameters, test_keys)
learning_rate = 0.001
flat_parameters_initial.requires_grad_(True)  # Enable gradient computation
optimizer = optim.Adam([flat_parameters_initial], lr=learning_rate)

losses = []


def closure():
    optimizer.zero_grad()
    result = objective_function(flat_parameters_initial, inducing_points_num, loaded_lambda_hyp, loaded_hyperparameters,
                                test_keys)
    total_loss = result[0]
    individual_fm_losses = result[1:]
    total_loss.backward()
    #


    hyperparameters = reconstruct_hyperparameters_test(
        flat_parameters_initial, loaded_hyperparameters, test_keys
    )



    losses.append((total_loss.item(),) + tuple(fm_loss.item() for fm_loss in individual_fm_losses))

    return total_loss


########################################################################################################################


# Initialize timing variables
start_time = time.time()

num_iterations = 50000
header_written = False
now = datetime.now()
formatted_now = now.strftime("%Y-%m-%d_%H-%M-%S")
main_folder = base_dir
file_name = f"optim_test_hyperparams_75_final"
main_folder_path = os.path.join(main_folder, f"{file_name}_{formatted_now}")
os.makedirs(main_folder_path, exist_ok=True)

for iteration in range(num_iterations):
    optimizer.step(closure)

    header_written = save_and_print_gp_hyperparams_test(
        iteration=iteration,
        num_iterations=num_iterations,
        losses=losses,
        flat_parameters_initial=flat_parameters_initial,
        loaded_hyperparameters=loaded_hyperparameters,
        test_keys=test_keys,
        optimized_lambda_hyp=loaded_lambda_hyp,
        main_folder_path=main_folder_path,
        file_name=file_name,
        save_interval=5000,
        print_interval=100,
        header_written=header_written,
        failure_modes=failure_modes,
        start_time=start_time,
    )

