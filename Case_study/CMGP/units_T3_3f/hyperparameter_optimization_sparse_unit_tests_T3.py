import torch.optim as optim
from datetime import datetime
from utils.utils_distinct_gp_per_unit.plot_save_print_unit import *
from utils.utils_distinct_gp_per_unit.data_processing_unit import *
from utils.utils_distinct_gp_per_unit.CMGP_unit import *

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
observed_data = pd.read_csv(os.path.join(base_dir, "historical_data.csv"))

train_file = "multi_mode_train_data.csv"
test_file = "multi_modes_test_data.csv"

sensors_list = ['T3']

time_limit = 10

unique_units = observed_data['unit number'].unique()
unique_failure_modes = observed_data['failure mode'].unique()

replicated_data = []

for unit in unique_units:
    # Filter data for the specific unit and the first time_limit points
    unit_data = observed_data[(observed_data['unit number'] == unit) & (observed_data['time, in cycles'] <= time_limit)]

    # Replicate this data for each failure mode
    for fm in unique_failure_modes:
        # Create a copy and set the failure mode
        replicated_unit_data = unit_data.copy()
        replicated_unit_data['failure mode'] = fm

        # Append to the list of replicated data
        replicated_data.append(replicated_unit_data)

replicated_data_df = pd.concat(replicated_data, ignore_index=True)

replicated_data_df['unit number'] = replicated_data_df['unit number'] + 1000

historical_test = pd.concat([observed_data, replicated_data_df], ignore_index=True)
test_data = replicated_data_df

data_dicts, all_signals, _ = create_data_dicts(historical_test, sensors_list)
data_dicts_test, _, _ = create_data_dicts(test_data, sensors_list)

failure_modes = observed_data['failure mode'].unique()

########################################################################################################################
# hyperparameters
loaded_hyperparams = torch.load(os.path.join(base_dir, r"optimized_hyperparameters_2024-09-03_14-47-48"
                                                       r"/iteration_50000/optimized_hyperparameters_50000.pth"))

# Accessing the loaded hyperparameters
loaded_hyperparameters = loaded_hyperparams['optimized_hyperparameters']
loaded_lambda_hyp = loaded_hyperparams['optimized_lambda_hyp']

initial_hyperparameters_test, _ = generate_initial_hyperparameters(data_dicts_test)
test_keys = initial_hyperparameters_test.keys()

print()


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

flat_parameters_initial.requires_grad_(True)  # Enable gradient computation
optimizer = optim.Adam([flat_parameters_initial], lr=0.001)

losses = []


def closure():
    optimizer.zero_grad()
    result = objective_function(flat_parameters_initial, inducing_points_num, loaded_lambda_hyp, loaded_hyperparameters,
                                test_keys)
    total_loss = result[0]
    individual_fm_losses = result[1:]
    total_loss.backward()

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
file_name = f"optimized_test_hyperparameters{sensors_list[0]}_{time_limit}"
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
