import neptune
from neptune.utils import stringify_unsupported
import torch.optim as optim
from datetime import datetime
from utils.utils_distinct_gp_per_unit.plot_save_print_unit import *
from utils.utils_distinct_gp_per_unit.data_processing_unit import *
from utils.utils_distinct_gp_per_unit.CMGP_unit import *
from collections import defaultdict


########################################################################################################################
t = 50
########################################################################################################################
inducing_points_num = 128
########################################################################################################################

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

########################################################################################################################
base_dir = os.path.dirname(os.path.abspath(__file__))
########################################################################################################################

historical_test = pd.read_csv(os.path.join(base_dir, f"historical_plus_test_data_{t}.csv"))
test_data = pd.read_csv(os.path.join(base_dir, f"test_data_{t}.csv"))

historical_test['time, in cycles'] = historical_test['time, in cycles'].astype('float32')
test_data['time, in cycles'] = test_data['time, in cycles'].astype('float32')

sensors_list = ['T42']

data_dicts, all_signals, _ = create_data_dicts(historical_test, sensors_list)
data_dicts_test, _, _ = create_data_dicts(test_data, sensors_list)

failure_modes = historical_test['failure mode'].unique()

########################################################################################################################

loaded_hyperparams = torch.load(
    os.path.join(base_dir, r"optimized_hyperparameters_2024-09-03_14-49-48/iteration_50000"
                           r"/optimized_hyperparameters_50000.pth"), map_location=device)

loaded_hyperparameters = loaded_hyperparams['optimized_hyperparameters']
loaded_lambda_hyp = loaded_hyperparams['optimized_lambda_hyp']


grouped_data = defaultdict(list)
for key, values in loaded_hyperparameters.items():
    group_key = key[1:]
    grouped_data[group_key].append(values)

loaded_hyperparameters_average = {}
for group_key, group_values in grouped_data.items():
    avg_values = {}
    for param in group_values[0].keys():
        avg_values[param] = torch.mean(torch.stack([v[param] for v in group_values]), dim=0)
    loaded_hyperparameters_average[group_key] = avg_values



initial_hyperparameters_test, _ = generate_initial_hyperparameters(data_dicts_test)
test_keys = initial_hyperparameters_test.keys()

for test_key in test_keys:
    sensor = test_key[1]
    mode = test_key[2]
    average_key = (sensor, mode)
    if average_key in loaded_hyperparameters_average:
        initial_hyperparameters_test[test_key] = loaded_hyperparameters_average[average_key]
    else:
        print(f"Warning: No match found for test key {test_key} in loaded_hyperparameters_average")


print()


########################################################################################################################

def objective_function(flat_parameters, number_of_inducing_points, lambda_hyp, hist_hyp, test_keys):
    hyperparameters = reconstruct_hyperparameters_test(flat_parameters, hist_hyp, test_keys)

    nll = {}
    for failure_mode, data_dict in data_dicts.items():

        current_lambda_hyp = lambda_hyp[failure_mode]

        all_sensor_readings = all_signals.get(failure_mode)

        nll_fm = neg_log_likelihood(all_sensor_readings, data_dict, hyperparameters, current_lambda_hyp,
                                    number_of_inducing_points, device)
        nll[failure_mode] = nll_fm

    total_nll = sum(nll.values())

    individual_nlls = [nll[fm] for fm in failure_modes]

    return total_nll, *individual_nlls


########################################################################################################################

# run = neptune.init_run(
#     project="TAMU/CS-gp",
#     api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJlYTNmMGJiNy00ZGViLTQwNjQtOTliMS1hYWU4YjRmMTI2Y2IifQ==",
#
# )
#
# def log_parameters_to_neptune(parameters):
#     for param_name, param_value in parameters.items():
#         if isinstance(param_value, torch.Tensor):
#             param_value = param_value.detach().cpu().numpy()
#             if param_value.size == 1:
#                 param_value = float(param_value.item())
#             else:
#                 for idx, val in enumerate(param_value.flatten()):
#                     run[f"parameters/{param_name}_{idx}"].append(
#                         { "value": float(val)}
#                     )
#                 continue
#         elif isinstance(param_value, list):
#             if len(param_value) == 1:
#                 param_value = param_value[0]
#             else:
#                 for idx, val in enumerate(param_value):
#                     run[f"parameters/{param_name}_{idx}"].append(
#                         {"value": val}
#                     )
#                 continue
#         elif not isinstance(param_value, (str, float, int)):
#             param_value = stringify_unsupported(param_value)
#
#         run[f"parameters/{param_name}"].append(
#             {"value": param_value}
#         )
#
#



flat_parameters_initial = flatten_hyperparameters_test(initial_hyperparameters_test)


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
    # run["train/total_loss"].append(total_loss.item())
    # for i, fm_loss in enumerate(individual_fm_losses):
    #     run[f"train/fm_loss_{i + 1}"].append(fm_loss.item())
    #
    # hyperparameters = reconstruct_hyperparameters_test(
    #     flat_parameters_initial, loaded_hyperparameters, test_keys
    # )
    #
    # log_parameters_to_neptune(hyperparameters)

    losses.append((total_loss.item(),) + tuple(fm_loss.item() for fm_loss in individual_fm_losses))

    return total_loss


########################################################################################################################


# Initialize timing variables
start_time = time.time()

num_iterations = 25000
header_written = False
now = datetime.now()
formatted_now = now.strftime("%Y-%m-%d_%H-%M-%S")
main_folder = base_dir
file_name = f"optim_test_hyperparams_{t}"
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
# run.stop()
