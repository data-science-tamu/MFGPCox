import neptune
from neptune.utils import stringify_unsupported
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
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

########################################################################################################################
base_dir = os.path.dirname(os.path.abspath(__file__))
########################################################################################################################
# Data
# Data
observed_data = pd.read_csv(os.path.join(base_dir, "historical_data.csv"))
observed_data['time, in cycles'] = observed_data['time, in cycles'].astype('float32')

sensors_list = ["sensor 2"]

data_dicts, all_signals, _ = create_data_dicts(observed_data, sensors_list)

failure_modes = observed_data['failure mode'].unique()
########################################################################################################################
# hyperparameters


# initial_hyperparameters, initial_lambda_hyp = generate_initial_hyperparameters_2(data_dicts)

loaded_hyperparams = torch.load(
    os.path.join(base_dir, r"optimized_hyperparameters_unit_specific_historical_res_2024-12-29_21-53-01"
                           r"/iteration_100000/optimized_hyperparameters_unit_specific_historical_res_100000.pth"),
    map_location=device)

# Accessing the loaded hyperparameters
loaded_hyperparameters = loaded_hyperparams['optimized_hyperparameters']
loaded_lambda_hyp = loaded_hyperparams['optimized_lambda_hyp']


initial_hyperparameters, initial_lambda_hyp = loaded_hyperparameters, loaded_lambda_hyp
########################################################################################################################


def objective_function(flat_parameters, number_of_inducing_points, template):
    hyperparameters, lambda_hyp = reconstruct_hyperparameters(flat_parameters, template, failure_modes)

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

flat_parameters_initial = flatten_hyperparameters(initial_hyperparameters, initial_lambda_hyp)
flat_parameters_initial.requires_grad_(True)

learning_rate = 0.001
optimizer = optim.Adam([flat_parameters_initial], lr=learning_rate)
losses = []


def closure():
    optimizer.zero_grad()
    result = objective_function(flat_parameters_initial, inducing_points_num, initial_hyperparameters)
    total_loss = result[0]
    individual_fm_losses = result[1:]
    total_loss.backward()

    losses.append((total_loss.item(),) + tuple(fm_loss.item() for fm_loss in individual_fm_losses))

    return total_loss


########################################################################################################################


# Initialize timing variables
start_time = time.time()

num_iterations = 200000
header_written = False
now = datetime.now()
formatted_now = now.strftime("%Y-%m-%d_%H-%M-%S")
main_folder = base_dir
file_name = "optimized_hyperparameters_unit_specific_historical_res2"
main_folder_path = os.path.join(main_folder, f"{file_name}_{formatted_now}")
os.makedirs(main_folder_path, exist_ok=True)

for iteration in range(num_iterations):
    optimizer.step(closure)

    header_written = save_and_print_gp_hyperparams(
        iteration=iteration,
        num_iterations=num_iterations,
        losses=losses,
        flat_parameters_initial=flat_parameters_initial,
        initial_hyperparameters=initial_hyperparameters,
        main_folder_path=main_folder_path,
        file_name=file_name,
        save_interval=5000,
        print_interval=100,
        header_written=header_written,
        failure_modes=failure_modes,
        start_time=start_time,
    )

