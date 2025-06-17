import torch.optim as optim
from datetime import datetime
from utils.utils_distinct_gp_per_unit.plot_save_print_unit import *
from utils.utils_distinct_gp_per_unit.data_processing_unit import *
from utils.utils_distinct_gp_per_unit.CMGP_unit import *

########################################################################################################################

# seed_value = 42
# torch.manual_seed(seed_value)
inducing_points_num = 500

########################################################################################################################

# specifying the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

########################################################################################################################
base_dir = os.path.dirname(os.path.abspath(__file__))
########################################################################################################################
# Data
# observed_data = pd.read_csv(os.path.join(base_dir, "historical_plus_test_data.csv"))

train_file = "multi_mode_train_data.csv"
test_file = "multi_modes_test_data.csv"

symbols = ["NL", "NH", "P13", "P26", "T26", "P3", "T3", "T6", "EPR", "T13", "P42", "T42", "P5", "T41", "Thrust", "Wf"]

failure_mode_ranges = {
    1: (range(1, 11), range(11, 16)),
    2: (range(31, 41), range(41, 46)),
    3: (range(101, 111), range(111, 116)),
}

output_filenames = {
    'historical': "historical_data.csv",
    'historical_plus_test': "historical_plus_test_data.csv",
}

sensors_list = ['P13']

observed_data, _ = process_datasets(base_dir, train_file, test_file, symbols, failure_mode_ranges, sensors_list,
                                    output_filenames)


data_dicts, all_signals, _ = create_data_dicts(observed_data, sensors_list)

failure_modes = observed_data['failure mode'].unique()
########################################################################################################################
# hyperparameters

initial_hyperparameters, initial_lambda_hyp = generate_initial_hyperparameters(data_dicts)


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

flat_parameters_initial.requires_grad_(True)  # Enable gradient computation
optimizer = optim.Adam([flat_parameters_initial], lr=0.001)

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

num_iterations = 1000000
header_written = False
now = datetime.now()
formatted_now = now.strftime("%Y-%m-%d_%H-%M-%S")
main_folder = base_dir
file_name = "optimized_hyperparameters_500_inducing_points_unit"
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
