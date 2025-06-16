import pandas as pd
import os
from utils.utils_final_fm.options_fm import *
########################################################################################################################

def process_datasets(base_dir, train_file, test_file, symbols, failure_mode_ranges, sensors_list, output_filenames):
    # Load train and test datasets
    train_df = pd.read_csv(os.path.join(base_dir, train_file), header=None)
    test_df = pd.read_csv(os.path.join(base_dir, test_file), header=None)

    # Define the column names
    new_columns = ["failure mode", "unit number", "time, in cycles"] + symbols
    train_df.columns = new_columns
    test_df.columns = new_columns

    # Define a function to filter datasets based on failure mode and unit number range
    def filter_dataset(train_df, test_df, failure_mode, train_range, test_range):
        train_mode = train_df[train_df['failure mode'] == failure_mode]
        test_mode = test_df[test_df['failure mode'] == failure_mode]
        train_subset = train_mode[train_mode['unit number'].isin(train_range)]
        test_subset = test_mode[test_mode['unit number'].isin(test_range)]
        return train_subset, test_subset

    # Apply the function to each failure mode and combine the subsets
    train_subsets = []
    test_subsets = []

    for mode, (train_range, test_range) in failure_mode_ranges.items():
        train_subset, test_subset = filter_dataset(train_df, test_df, mode, train_range, test_range)
        train_subsets.append(train_subset)
        test_subsets.append(test_subset)

    # Combine all train and test subsets
    train = pd.concat(train_subsets)
    test = pd.concat(test_subsets)

    # Columns to include in the final datasets
    columns_to_include = ['failure mode', 'unit number', 'time, in cycles'] + sensors_list

    # Create the final train DataFrame
    historical_data = train[columns_to_include].reset_index(drop=True)
    historical_data.to_csv(os.path.join(base_dir, output_filenames['historical']), index=False)

    # Combine the train and test DataFrames
    combined_df = pd.concat([train, test])

    # Create the final combined DataFrame
    historical_plus_test_data = combined_df[columns_to_include].reset_index(drop=True)
    historical_plus_test_data.to_csv(os.path.join(base_dir, output_filenames['historical_plus_test']), index=False)

    return historical_plus_test_data, historical_data




########################################################################################################################
def create_data_dicts(observed_data, sensors):
    """
    This function creates a dictionary for each combination of sensor and failure mode,
    where the keys are tuples of (unit, sensor, failure_mode), and the values are dictionaries containing
    time points and sensor readings. It also returns a list of sensor readings and time points
    specific to each combination of sensor and failure mode.

    Parameters:
    observed_data (pd.DataFrame): The dataset containing the data.
    sensors (list of str): A list of sensor names to be used for filtering and dictionary creation.

    Returns:
    dict: A dictionary where each key is a combination of (sensor, failure_mode) and the value
          is another dictionary containing filtered data for each unit.
    dict: A dictionary containing all sensor readings specific to each (sensor, failure_mode) combination.
    dict: A dictionary containing all time points specific to each (sensor, failure_mode) combination.
    """

    failure_modes = observed_data['failure mode'].unique()
    all_data_dicts = {}
    all_sensor_readings_dict = {}
    all_time_points_dict = {}

    # Loop through each combination of sensor and failure mode
    for sensor in sensors:
        for failure_mode in failure_modes:
            data_dict = {}
            all_sensor_readings = []
            all_time_points = []

            # Loop through each unit number
            for unit in observed_data['unit number'].unique():
                # Filter the dataframe for this specific combination of unit, sensor, and failure mode
                subset_df = observed_data[(observed_data['unit number'] == unit)
                                          & (observed_data['failure mode'] == failure_mode)]

                # Check if the subset is not empty
                if not subset_df.empty:
                    key = (unit, sensor, failure_mode)
                    sensor_readings = subset_df[sensor].values
                    data_dict[key] = {
                        'time_points': subset_df['time, in cycles'].values,
                        'sensor_readings': sensor_readings
                    }
                    # Add the sensor readings and time points to the combined list for this sensor and failure mode
                    all_sensor_readings.extend(sensor_readings)
                    all_time_points.extend(subset_df['time, in cycles'].values)

            # Add the dictionary and sensor readings for this sensor and failure mode combination to the overall dictionaries
            combined_key = (sensor, failure_mode)
            all_data_dicts[combined_key] = data_dict
            all_sensor_readings_dict[combined_key] = all_sensor_readings
            all_time_points_dict[combined_key] = all_time_points

    return all_data_dicts, all_sensor_readings_dict, all_time_points_dict



########################################################################################################################
def generate_initial_hyperparameters(data_dicts, preferred_device=device):
    initial_hyperparameters = {}
    initial_lambda_hyp = {}

    # Iterate over all data dictionaries in data_dicts
    for (sensor, failure_mode), data_dict in data_dicts.items():
        # Initialize lambda for this sensor and failure mode combination
        initial_lambda_hyp[(sensor, failure_mode)] = torch.rand(1, device=preferred_device)

        for (unit, sensor, fm) in data_dict.keys():
            # Check if the combination of unit, sensor, and failure mode has already been processed
            if (sensor, fm) not in initial_hyperparameters:
                # Generate hyperparameters for this unique combination
                initial_hyperparameters[(sensor, fm)] = {
                    'alpha': torch.rand(1, device=preferred_device) + 1,
                    'xi': torch.rand(1, device=preferred_device) + 1,
                    'sigma': torch.normal(mean=0, std=0.0001, size=(1,), device=preferred_device)
                }

    return initial_hyperparameters, initial_lambda_hyp


########################################################################################################################
# flatting hyperparameters from dict to an array
# def flatten_hyperparameters(hyperparameters, lambda_hyp):
#     flat_parameters = []
#
#     # Flatten lambda_hyp for each failure mode
#     for fm, lambda_val in lambda_hyp.items():
#         flat_parameters.append(lambda_val)
#
#     # Flatten the hyperparameters
#     for key, value in hyperparameters.items():
#         flat_parameters.extend([value['alpha'], value['xi'], value['sigma']])
#
#     return torch.cat(flat_parameters)


def flatten_hyperparameters(hyperparameters, lambda_hyp):
    flat_parameters = []

    # Flatten lambda_hyp for each (sensor, failure_mode) combination
    for (sensor, fm), lambda_val in lambda_hyp.items():
        flat_parameters.append(lambda_val)

    # Flatten the hyperparameters for each (unit, sensor, fm) combination
    for key, value in hyperparameters.items():
        flat_parameters.extend([value['alpha'], value['xi'], value['sigma']])

    return torch.cat(flat_parameters)


# Reconstruct hyperparameters from a flat array
# def reconstruct_hyperparameters(flat_parameters, template, failure_modes):
#     lambda_hyp = {}
#     hyperparameters = {}
#     index = 0
#
#     # Reconstruct lambda_hyp for each failure mode
#     for fm in failure_modes:
#         lambda_hyp[fm] = flat_parameters[index].unsqueeze(0)
#         index += 1
#
#     # Reconstruct the hyperparameters
#     for key in template.keys():
#         hyperparameters[key] = {
#             'alpha': flat_parameters[index].unsqueeze(0),
#             'xi': flat_parameters[index + 1].unsqueeze(0),
#             'sigma': flat_parameters[index + 2].unsqueeze(0),
#         }
#         index += 3
#
#     return hyperparameters, lambda_hyp

def reconstruct_hyperparameters(flat_parameters, template, failure_modes_sensors):
    lambda_hyp = {}
    hyperparameters = {}
    index = 0

    # Reconstruct lambda_hyp for each (sensor, failure_mode) combination
    for key in failure_modes_sensors:
        lambda_hyp[key] = flat_parameters[index].unsqueeze(0)
        index += 1

    # Reconstruct the hyperparameters for each (unit, sensor, fm) combination
    for key in template.keys():
        hyperparameters[key] = {
            'alpha': flat_parameters[index].unsqueeze(0),
            'xi': flat_parameters[index + 1].unsqueeze(0),
            'sigma': flat_parameters[index + 2].unsqueeze(0),
        }
        index += 3

    return hyperparameters, lambda_hyp


########################################################################################################################
def flatten_hyperparameters_elbo(pi_hat, b, psi, alpha_hat, beta, gamma):
    # Convert dictionaries to tensors
    tensors = []
    metadata = []

    for name, param_dict in [('pi_hat', pi_hat), ('b', b),
                             ('psi', psi), ('alpha_hat', alpha_hat),
                             ('beta', beta), ('gamma', gamma)]:
        keys, values = zip(*param_dict.items())
        tensor = torch.tensor(values, dtype=torch.float32)
        tensors.append(tensor)
        metadata.append((name, keys, tensor.shape))

    flat_parameters = torch.cat(tensors)
    return flat_parameters, metadata


# flat_parameters_initial, metadata = flatten_hyperparameters(initial_pi_hat, initial_b, initial_psi, initial_alpha_hat,
#                                                             initial_beta, initial_gamma)

def reconstruct_hyperparameters_elbo(flat_parameters, metadata):
    reconstructed_params = {}
    offset = 0
    for name, keys, shape in metadata:
        length = torch.prod(torch.tensor(shape)).item()  # Calculate the number of elements
        values = flat_parameters[offset:offset + length].reshape(shape)

        # if name == 'alpha_hat':
        #     # Apply exponential transformation to ensure values are positive
        #     values = torch.exp(values)
        # elif name == 'pi_hat':
        #     # Apply softmax to ensure values are positive and sum to 1
        #     # values = torch.softmax(values, dim=0)
        #     values = torch.exp(values)
        reconstructed_params[name] = {key: val for key, val in zip(keys, values)}
        offset += length

    return reconstructed_params



########################################################################################################################

def flatten_hyperparameters_elbo_2(pi_hat, mu_b_hat, sigma_b_hat, alpha_rho_hat, beta_rho_hat, alpha_hat, beta, gamma):
    tensors = []
    metadata = []

    for name, param_dict in [('pi_hat', pi_hat), ('mu_b_hat', mu_b_hat),
                             ('sigma_b_hat', sigma_b_hat), ('alpha_rho_hat', alpha_rho_hat),
                             ('beta_rho_hat', beta_rho_hat), ('alpha_hat', alpha_hat),
                             ('beta', beta), ('gamma', gamma)]:
        keys, values = zip(*param_dict.items())
        tensor = torch.tensor(values, dtype=torch.float32)
        tensors.append(tensor)
        metadata.append((name, keys, tensor.shape))

    flat_parameters = torch.cat(tensors)
    return flat_parameters, metadata



def reconstruct_hyperparameters_elbo_2(flat_parameters, metadata):
    reconstructed_params = {}
    offset = 0
    for name, keys, shape in metadata:
        length = torch.prod(torch.tensor(shape)).item()
        values = flat_parameters[offset:offset + length].reshape(shape)

        if name in ['alpha_rho_hat', 'beta_rho_hat']:
            values = torch.exp(values)

        reconstructed_params[name] = {key: val for key, val in zip(keys, values)}
        offset += length

    return reconstructed_params
