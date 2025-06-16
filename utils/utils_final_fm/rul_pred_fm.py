import torch
from utils.utils_final_fm.CMGP_fm import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def process_unit_data(train_data, all_data, device):
    # Initialize dictionaries for storing event times
    train_units_event_time = {}
    test_units_event_time = {}

    # Get unique unit numbers from train and all data
    unique_historical_units = train_data['unit number'].unique()
    unique_units = all_data['unit number'].unique()

    # Identify train units
    train_unit_range = unique_historical_units

    # Calculate max event time for each unit and categorize as train or test
    for unit in unique_units:
        subset_df = all_data[all_data['unit number'] == unit]
        max_event_time = subset_df['time, in cycles'].max()
        event_time_tensor = torch.tensor(max_event_time, dtype=torch.float32, device=device)
        if unit in train_unit_range:
            train_units_event_time[unit] = event_time_tensor
        else:
            test_units_event_time[unit] = event_time_tensor

    # Calculate minimum event time for each failure mode
    failure_mode_groups = train_data.groupby('failure mode')
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

    # Create dictionaries for unit status, manufacture status, and failure mode
    unit_status = {}
    unit_manufac = {}
    unit_failure_mode = {}

    for unit in unique_units:
        subset_df = all_data[all_data['unit number'] == unit]
        unit_status[unit] = torch.tensor(1, dtype=torch.float32, device=device)  # Default status as failure
        unit_manufac[unit] = torch.tensor(0, dtype=torch.float32, device=device)  # Default manufacture status
        failure = subset_df['failure mode'].max()
        unit_failure_mode[unit] = failure

    # Return all calculated dictionaries
    return (train_units_event_time, test_units_event_time, min_V_by_failure_mode,
            unit_status, unit_manufac, unit_failure_mode)


def likelihood(sensor_readings, predicted, cov, device=device):
    y = sensor_readings
    _, log_det = torch.linalg.slogdet(cov)
    log_likelihood = -0.5 * (y - predicted).transpose(0, 1) @ torch.linalg.solve(cov, (
            y - predicted)) - 0.5 * log_det - y.size(0) / 2 * torch.log(torch.tensor(2 * torch.pi, device=device))
    l = log_likelihood
    likelihood = torch.exp(l)
    return likelihood, l


def probability_of_failure(data_dicts, all_sensor_readings, loaded_hyperparameters, loaded_lambda_hyp,
                           approx_cov_results, unit, actual_failure, pi_hat, sensor):
    data_dict = data_dicts.get((sensor, actual_failure))
    historical_inputs = torch.tensor(data_dict[(unit, sensor, actual_failure)]['time_points'],
                                     device=device)

    historical_outputs = torch.tensor(data_dict[(unit, sensor, actual_failure)]['sensor_readings'],
                                      dtype=torch.float32,
                                      device=device).unsqueeze(-1)

    means_1, variance_1 = get_cmgp_predictions(all_sensor_readings, historical_inputs, inducing_points_num, unit,
                                               [sensor],
                                               1,
                                               data_dicts, loaded_hyperparameters,
                                               loaded_lambda_hyp, approx_cov_results, preferred_device=device)

    means_2, variance_2 = get_cmgp_predictions(all_sensor_readings, historical_inputs, inducing_points_num, unit,
                                               [sensor],
                                               2,
                                               data_dicts, loaded_hyperparameters,
                                               loaded_lambda_hyp, approx_cov_results, preferred_device=device)

    means_3, variance_3 = get_cmgp_predictions(all_sensor_readings, historical_inputs, inducing_points_num, unit,
                                               [sensor],
                                               3,
                                               data_dicts, loaded_hyperparameters,
                                               loaded_lambda_hyp, approx_cov_results, preferred_device=device)

    prob1, log_like1 = likelihood(historical_outputs.squeeze().unsqueeze(-1), means_1.squeeze().unsqueeze(-1),
                                  variance_1.squeeze())
    prob2, log_like2 = likelihood(historical_outputs.squeeze().unsqueeze(-1), means_2.squeeze().unsqueeze(-1),
                                  variance_2.squeeze())
    prob3, log_like3 = likelihood(historical_outputs.squeeze().unsqueeze(-1), means_3.squeeze().unsqueeze(-1),
                                  variance_3.squeeze())

    # print(f'unit:{unit} ==> log_likelihood_1: {log_like1.item()}, log_likelihood_2: {log_like2.item()}, '
    #       f'log_likelihood_3: {log_like3.item()}')
    # print(f'unit:{unit} ==> likelihood_1: {prob1.item()}, likelihood_2: {prob2.item()}, likelihood_3: {prob3.item()}')

    # Check if the maximum log-likelihood is greater than 88 and adjust if necessary
    max_log_likelihood = max(log_like1, log_like2, log_like3).item()

    # Check if the maximum log-likelihood is greater than 88 and adjust if necessary
    if max_log_likelihood > 88:
        subtract_value = max_log_likelihood - 88

        # Subtract the necessary value from all log-likelihoods
        log_like1 -= subtract_value
        log_like2 -= subtract_value
        log_like3 -= subtract_value

        # Recalculate the probabilities after adjustment
        prob1 = torch.exp(log_like1)
        prob2 = torch.exp(log_like2)
        prob3 = torch.exp(log_like3)
    # print(sensor)
    # print(f'unit:{unit} ==> Re_log_likelihood_1: {log_like1.item()}, Re_log_likelihood_2: {log_like2.item()}, '
    #       f'Re_log_likelihood_3: {log_like3.item()}')
    # print(
    #     f'unit:{unit} ==> Re_likelihood_1: {prob1.item()}, Re_likelihood_2: {prob2.item()}, Re_likelihood_3: {prob3.item()}')

    # print(f'unit:{unit} ==> likelihood_1: {prob1.item()}, likelihood_2: {prob2.item()}, likelihood_3: {prob3.item()}')

    pi1 = pi_hat.get(1)
    pi2 = pi_hat.get(2)
    pi3 = pi_hat.get(3)
    # n = len(historical_inputs)

    denominator = pi1 * prob1 + pi2 * prob2 + pi3 * prob3

    prob_one = (pi1 * prob1) / denominator
    prob_two = (pi2 * prob2) / denominator
    prob_three = (pi3 * prob3) / denominator

    # denominator = pi1**n * prob1 + pi2**n * prob2 + pi3**n * prob3
    #
    # prob_one = (pi1**n * prob1) / denominator
    # prob_two = (pi2**n * prob2) / denominator
    # prob_three = (pi3**n * prob3) / denominator

    # Check for NaN and replace with 1 if necessary
    prob_one = torch.where(torch.isnan(prob_one), torch.tensor(1.0), prob_one)
    prob_two = torch.where(torch.isnan(prob_two), torch.tensor(1.0), prob_two)
    prob_three = torch.where(torch.isnan(prob_three), torch.tensor(1.0), prob_three)

    return prob_one, prob_two, prob_three


def probability_of_failure_all_test(data_dict_test, data_dicts, all_sensor_readings, loaded_hyperparameters,
                                    loaded_lambda_hyp,
                                    approx_cov_results, unit, actual_failure, pi_hat, sensor):
    data_dict = data_dict_test.get((sensor, actual_failure))
    historical_inputs = torch.tensor(data_dict[(unit, sensor, actual_failure)]['time_points'],
                                     device=device)

    historical_outputs = torch.tensor(data_dict[(unit, sensor, actual_failure)]['sensor_readings'],
                                      dtype=torch.float32,
                                      device=device).unsqueeze(-1)

    means_1, variance_1 = get_cmgp_predictions(all_sensor_readings, historical_inputs, inducing_points_num, unit,
                                               [sensor],
                                               1,
                                               data_dicts, loaded_hyperparameters,
                                               loaded_lambda_hyp, approx_cov_results, preferred_device=device)

    means_2, variance_2 = get_cmgp_predictions(all_sensor_readings, historical_inputs, inducing_points_num, unit,
                                               [sensor],
                                               2,
                                               data_dicts, loaded_hyperparameters,
                                               loaded_lambda_hyp, approx_cov_results, preferred_device=device)

    means_3, variance_3 = get_cmgp_predictions(all_sensor_readings, historical_inputs, inducing_points_num, unit,
                                               [sensor],
                                               3,
                                               data_dicts, loaded_hyperparameters,
                                               loaded_lambda_hyp, approx_cov_results, preferred_device=device)

    prob1, log_like1 = likelihood(historical_outputs.squeeze().unsqueeze(-1), means_1.squeeze().unsqueeze(-1),
                                  variance_1.squeeze())
    prob2, log_like2 = likelihood(historical_outputs.squeeze().unsqueeze(-1), means_2.squeeze().unsqueeze(-1),
                                  variance_2.squeeze())
    prob3, log_like3 = likelihood(historical_outputs.squeeze().unsqueeze(-1), means_3.squeeze().unsqueeze(-1),
                                  variance_3.squeeze())

    # print(f'unit:{unit} ==> log_likelihood_1: {log_like1.item()}, log_likelihood_2: {log_like2.item()}, '
    #       f'log_likelihood_3: {log_like3.item()}')
    # print(f'unit:{unit} ==> likelihood_1: {prob1.item()}, likelihood_2: {prob2.item()}, likelihood_3: {prob3.item()}')

    max_log_likelihood = max(log_like1, log_like2, log_like3).item()

    # Check if the maximum log-likelihood is greater than 88 and adjust if necessary
    max_log_likelihood = max(log_like1, log_like2, log_like3).item()

    # Check if the maximum log-likelihood is greater than 88 and adjust if necessary
    if max_log_likelihood > 88:
        subtract_value = max_log_likelihood - 88

        # Subtract the necessary value from all log-likelihoods
        log_like1 -= subtract_value
        log_like2 -= subtract_value
        log_like3 -= subtract_value

        # Recalculate the probabilities after adjustment
        prob1 = torch.exp(log_like1)
        prob2 = torch.exp(log_like2)
        prob3 = torch.exp(log_like3)
    # print(sensor)
    # print(f'unit:{unit} ==> Re_log_likelihood_1: {log_like1.item()}, Re_log_likelihood_2: {log_like2.item()}, '
    #       f'Re_log_likelihood_3: {log_like3.item()}')
    # print(
    #     f'unit:{unit} ==> Re_likelihood_1: {prob1.item()}, Re_likelihood_2: {prob2.item()}, Re_likelihood_3: {prob3.item()}')

    # print(f'unit:{unit} ==> likelihood_1: {prob1.item()}, likelihood_2: {prob2.item()}, likelihood_3: {prob3.item()}')

    pi1 = pi_hat.get(1)
    pi2 = pi_hat.get(2)
    pi3 = pi_hat.get(3)
    # n = len(historical_inputs)

    denominator = pi1 * prob1 + pi2 * prob2 + pi3 * prob3

    prob_one = (pi1 * prob1) / denominator
    prob_two = (pi2 * prob2) / denominator
    prob_three = (pi3 * prob3) / denominator

    # denominator = pi1**n * prob1 + pi2**n * prob2 + pi3**n * prob3
    #
    # prob_one = (pi1**n * prob1) / denominator
    # prob_two = (pi2**n * prob2) / denominator
    # prob_three = (pi3**n * prob3) / denominator

    # Check for NaN and replace with 1 if necessary
    prob_one = torch.where(torch.isnan(prob_one), torch.tensor(1.0), prob_one)
    prob_two = torch.where(torch.isnan(prob_two), torch.tensor(1.0), prob_two)
    prob_three = torch.where(torch.isnan(prob_three), torch.tensor(1.0), prob_three)

    return prob_one, prob_two, prob_three


def likelihood_of_failure_all_test(data_dict_test, data_dicts, all_sensor_readings, loaded_hyperparameters,
                                   loaded_lambda_hyp,
                                   approx_cov_results, unit, actual_failure, pi_hat, sensor):
    data_dict = data_dict_test.get((sensor, actual_failure))
    historical_inputs = torch.tensor(data_dict[(unit, sensor, actual_failure)]['time_points'],
                                     device=device)

    historical_outputs = torch.tensor(data_dict[(unit, sensor, actual_failure)]['sensor_readings'],
                                      dtype=torch.float32,
                                      device=device).unsqueeze(-1)

    means_1, variance_1 = get_cmgp_predictions(all_sensor_readings, historical_inputs, inducing_points_num, unit,
                                               [sensor],
                                               1,
                                               data_dicts, loaded_hyperparameters,
                                               loaded_lambda_hyp, approx_cov_results, preferred_device=device)

    means_2, variance_2 = get_cmgp_predictions(all_sensor_readings, historical_inputs, inducing_points_num, unit,
                                               [sensor],
                                               2,
                                               data_dicts, loaded_hyperparameters,
                                               loaded_lambda_hyp, approx_cov_results, preferred_device=device)

    means_3, variance_3 = get_cmgp_predictions(all_sensor_readings, historical_inputs, inducing_points_num, unit,
                                               [sensor],
                                               3,
                                               data_dicts, loaded_hyperparameters,
                                               loaded_lambda_hyp, approx_cov_results, preferred_device=device)

    prob1, log_like1 = likelihood(historical_outputs.squeeze().unsqueeze(-1), means_1.squeeze().unsqueeze(-1),
                                  variance_1.squeeze())
    prob2, log_like2 = likelihood(historical_outputs.squeeze().unsqueeze(-1), means_2.squeeze().unsqueeze(-1),
                                  variance_2.squeeze())
    prob3, log_like3 = likelihood(historical_outputs.squeeze().unsqueeze(-1), means_3.squeeze().unsqueeze(-1),
                                  variance_3.squeeze())

    # print(f'unit:{unit} ==> log_likelihood_1: {log_like1.item()}, log_likelihood_2: {log_like2.item()}, '
    #       f'log_likelihood_3: {log_like3.item()}')
    # print(f'unit:{unit} ==> likelihood_1: {prob1.item()}, likelihood_2: {prob2.item()}, likelihood_3: {prob3.item()}')

    max_log_likelihood = max(log_like1, log_like2, log_like3).item()

    # Check if the maximum log-likelihood is greater than 88 and adjust if necessary
    max_log_likelihood = max(log_like1, log_like2, log_like3).item()

    # Check if the maximum log-likelihood is greater than 88 and adjust if necessary
    if max_log_likelihood > 88:
        subtract_value = max_log_likelihood - 88

        # Subtract the necessary value from all log-likelihoods
        log_like1 -= subtract_value
        log_like2 -= subtract_value
        log_like3 -= subtract_value

        # Recalculate the probabilities after adjustment
        prob1 = torch.exp(log_like1)
        prob2 = torch.exp(log_like2)
        prob3 = torch.exp(log_like3)

    c = 1 / 1
    return c * prob1, c * prob2, c * prob3


def probability_of_failure_hist(slice_value, data_dicts, all_sensor_readings, loaded_hyperparameters, loaded_lambda_hyp,
                                approx_cov_results, unit, actual_failure, pi_hat, sensor):
    data_dict = data_dicts.get((sensor, actual_failure))
    historical_inputs = torch.tensor(data_dict[(unit, sensor, actual_failure)]['time_points'][:slice_value],
                                     device=device)

    historical_outputs = torch.tensor(data_dict[(unit, sensor, actual_failure)]['sensor_readings'][:slice_value],
                                      dtype=torch.float32,
                                      device=device).unsqueeze(-1)

    means_1, variance_1 = get_cmgp_predictions(all_sensor_readings, historical_inputs, inducing_points_num, unit,
                                               [sensor],
                                               1,
                                               data_dicts, loaded_hyperparameters,
                                               loaded_lambda_hyp, approx_cov_results, preferred_device=device)

    means_2, variance_2 = get_cmgp_predictions(all_sensor_readings, historical_inputs, inducing_points_num, unit,
                                               [sensor],
                                               2,
                                               data_dicts, loaded_hyperparameters,
                                               loaded_lambda_hyp, approx_cov_results, preferred_device=device)

    means_3, variance_3 = get_cmgp_predictions(all_sensor_readings, historical_inputs, inducing_points_num, unit,
                                               [sensor],
                                               3,
                                               data_dicts, loaded_hyperparameters,
                                               loaded_lambda_hyp, approx_cov_results, preferred_device=device)

    prob1, log_like1 = likelihood(historical_outputs.squeeze().unsqueeze(-1), means_1.squeeze().unsqueeze(-1),
                                  variance_1.squeeze())
    prob2, log_like2 = likelihood(historical_outputs.squeeze().unsqueeze(-1), means_2.squeeze().unsqueeze(-1),
                                  variance_2.squeeze())
    prob3, log_like3 = likelihood(historical_outputs.squeeze().unsqueeze(-1), means_3.squeeze().unsqueeze(-1),
                                  variance_3.squeeze())

    # print(f'unit:{unit} ==> log_likelihood_1: {log_like1.item()}, log_likelihood_2: {log_like2.item()}, '
    #       f'log_likelihood_3: {log_like3.item()}')
    # print(f'unit:{unit} ==> likelihood_1: {prob1.item()}, likelihood_2: {prob2.item()}, likelihood_3: {prob3.item()}')

    max_log_likelihood = max(log_like1, log_like2, log_like3).item()

    # Check if the maximum log-likelihood is greater than 88 and adjust if necessary
    max_log_likelihood = max(log_like1, log_like2, log_like3).item()

    # Check if the maximum log-likelihood is greater than 88 and adjust if necessary
    if max_log_likelihood > 88:
        subtract_value = max_log_likelihood - 88

        # Subtract the necessary value from all log-likelihoods
        log_like1 -= subtract_value
        log_like2 -= subtract_value
        log_like3 -= subtract_value

        # Recalculate the probabilities after adjustment
        prob1 = torch.exp(log_like1)
        prob2 = torch.exp(log_like2)
        prob3 = torch.exp(log_like3)
    # print(sensor)
    # print(f'unit:{unit} ==> Re_log_likelihood_1: {log_like1.item()}, Re_log_likelihood_2: {log_like2.item()}, '
    #       f'Re_log_likelihood_3: {log_like3.item()}')
    # print(
    #     f'unit:{unit} ==> Re_likelihood_1: {prob1.item()}, Re_likelihood_2: {prob2.item()}, Re_likelihood_3: {prob3.item()}')

    # print(f'unit:{unit} ==> likelihood_1: {prob1.item()}, likelihood_2: {prob2.item()}, likelihood_3: {prob3.item()}')

    pi1 = pi_hat.get(1)
    pi2 = pi_hat.get(2)
    pi3 = pi_hat.get(3)
    # n = len(historical_inputs)

    denominator = pi1 * prob1 + pi2 * prob2 + pi3 * prob3

    prob_one = (pi1 * prob1) / denominator
    prob_two = (pi2 * prob2) / denominator
    prob_three = (pi3 * prob3) / denominator

    # denominator = pi1**n * prob1 + pi2**n * prob2 + pi3**n * prob3
    #
    # prob_one = (pi1**n * prob1) / denominator
    # prob_two = (pi2**n * prob2) / denominator
    # prob_three = (pi3**n * prob3) / denominator

    # Check for NaN and replace with 1 if necessary
    prob_one = torch.where(torch.isnan(prob_one), torch.tensor(1.0), prob_one)
    prob_two = torch.where(torch.isnan(prob_two), torch.tensor(1.0), prob_two)
    prob_three = torch.where(torch.isnan(prob_three), torch.tensor(1.0), prob_three)

    return prob_one, prob_two, prob_three


def likelihood_of_failure_hist(slice_value, data_dicts, all_sensor_readings, loaded_hyperparameters, loaded_lambda_hyp,
                               approx_cov_results, unit, actual_failure, pi_hat, sensor):
    data_dict = data_dicts.get((sensor, actual_failure))
    historical_inputs = torch.tensor(data_dict[(unit, sensor, actual_failure)]['time_points'][:slice_value],
                                     device=device)

    historical_outputs = torch.tensor(data_dict[(unit, sensor, actual_failure)]['sensor_readings'][:slice_value],
                                      dtype=torch.float32,
                                      device=device).unsqueeze(-1)

    means_1, variance_1 = get_cmgp_predictions(all_sensor_readings, historical_inputs, inducing_points_num, unit,
                                               [sensor],
                                               1,
                                               data_dicts, loaded_hyperparameters,
                                               loaded_lambda_hyp, approx_cov_results, preferred_device=device)

    means_2, variance_2 = get_cmgp_predictions(all_sensor_readings, historical_inputs, inducing_points_num, unit,
                                               [sensor],
                                               2,
                                               data_dicts, loaded_hyperparameters,
                                               loaded_lambda_hyp, approx_cov_results, preferred_device=device)

    means_3, variance_3 = get_cmgp_predictions(all_sensor_readings, historical_inputs, inducing_points_num, unit,
                                               [sensor],
                                               3,
                                               data_dicts, loaded_hyperparameters,
                                               loaded_lambda_hyp, approx_cov_results, preferred_device=device)

    prob1, log_like1 = likelihood(historical_outputs.squeeze().unsqueeze(-1), means_1.squeeze().unsqueeze(-1),
                                  variance_1.squeeze())
    prob2, log_like2 = likelihood(historical_outputs.squeeze().unsqueeze(-1), means_2.squeeze().unsqueeze(-1),
                                  variance_2.squeeze())
    prob3, log_like3 = likelihood(historical_outputs.squeeze().unsqueeze(-1), means_3.squeeze().unsqueeze(-1),
                                  variance_3.squeeze())

    # print(f'unit:{unit} ==> log_likelihood_1: {log_like1.item()}, log_likelihood_2: {log_like2.item()}, '
    #       f'log_likelihood_3: {log_like3.item()}')
    # print(f'unit:{unit} ==> likelihood_1: {prob1.item()}, likelihood_2: {prob2.item()}, likelihood_3: {prob3.item()}')

    max_log_likelihood = max(log_like1, log_like2, log_like3).item()

    # Check if the maximum log-likelihood is greater than 88 and adjust if necessary
    max_log_likelihood = max(log_like1, log_like2, log_like3).item()

    # Check if the maximum log-likelihood is greater than 88 and adjust if necessary
    if max_log_likelihood > 88:
        subtract_value = max_log_likelihood - 88

        # Subtract the necessary value from all log-likelihoods
        log_like1 -= subtract_value
        log_like2 -= subtract_value
        log_like3 -= subtract_value

        # Recalculate the probabilities after adjustment
        prob1 = torch.exp(log_like1)
        prob2 = torch.exp(log_like2)
        prob3 = torch.exp(log_like3)

    return prob1, prob2, prob3


def h0(l, b, psi, min_V):
    return torch.where(l >= min_V, torch.exp(b + psi ** 2 * (l - min_V)), torch.tensor(0.0))


# fm spicific
def integrand(all_sensor_readings, approx_cov_results, unit_manufac, sensors, l, b, psi, beta, gamma, unit,
              unit_failure, min_V, data, hyperparameters, lambda_hyperparameter):
    h0_value = h0(l, b, psi, min_V)
    # means, variances = get_cmgp_predictions(l, unit, 'P13', 1, 1,
    #                                         data, hyperparameters,
    #                                         lambda_hyperparameter, sigma_hyperparameter, a,
    #                                         failure_mode_specific=False, preferred_device=device)

    means, _ = get_cmgp_predictions(all_sensor_readings, l, inducing_points_num, unit,
                                    sensors,
                                    unit_failure,
                                    data, hyperparameters,
                                    lambda_hyperparameter, approx_cov_results, preferred_device=device)

    means = means.squeeze()
    if beta.shape == torch.Size([1]):
        exp_component_first = beta * means
    else:
        exp_component_first = beta @ means
    # exp_component_third = gamma * unit_manufac.get(unit)
    exp_component_third = torch.tensor(0.0, device=device)
    exp_component = exp_component_first + exp_component_third

    return h0_value * torch.exp(exp_component)


#
# def integrand(all_sensor_readings, approx_cov_results, unit_manufac, sensors, l, b, psi, beta, gamma, unit,
#               unit_failure, min_V, data, hyperparameters, lambda_hyperparameter, unit_f, loaded_hyperparameters_fm,
#               loaded_lambda_hyp_fm):
#     h0_value = h0(l, b, psi, min_V)
#
#     # means, variances = get_cmgp_predictions(l, unit, 'P13', 1, 1,
#     #                                         data, hyperparameters,
#     #                                         lambda_hyperparameter, sigma_hyperparameter, a,
#     #                                         failure_mode_specific=False, preferred_device=device)
#     if (unit, unit_failure) in unit_f:
#         means, _ = get_cmgp_predictions(all_sensor_readings, l, inducing_points_num, unit,
#                                         sensors,
#                                         unit_failure,
#                                         data, hyperparameters,
#                                         lambda_hyperparameter, approx_cov_results, preferred_device=device)
#     else:
#         means, _ = get_cmgp_predictions_fm(all_sensor_readings, l, inducing_points_num, unit,
#                                            sensors,
#                                            unit_failure,
#                                            data, loaded_hyperparameters_fm,
#                                            loaded_lambda_hyp_fm, approx_cov_results, preferred_device=device)
#
#     means = means.squeeze()
#     if beta.shape == torch.Size([1]):
#         exp_component_first = beta * means
#     else:
#         exp_component_first = beta @ means
#     # exp_component_third = gamma * unit_manufac.get(unit)
#     exp_component_third = torch.tensor(0.0, device=device)
#     exp_component = exp_component_first + exp_component_third
#
#     return h0_value * torch.exp(exp_component)


def St_cond_EST(min_V_by_failure_mode, unit_f_probabilities, failure_modes, all_sensor_readings, approx_cov_results,
                unit_manufac, sensors, unit, t_star, t, b_dict, psi_dict, beta_dict, gamma_dict, data, hyperparameters,
                lambda_hyperparameter):
    s = {}
    for failure in failure_modes:

        p = unit_f_probabilities.get(unit).get(failure)
        b = b_dict.get(failure)
        psi = psi_dict.get(failure)
        # beta = beta_dict.get(unit_failure)
        beta_values = []
        for sensor in sensors:
            beta_value = beta_dict.get((sensor, failure))
            if beta_value is not None:
                beta_values.append(beta_value)
            else:
                raise ValueError(f"Beta value for sensor {sensor} and failure mode {failure} not found.")
        beta = torch.tensor(beta_values, device=device)
        gamma = gamma_dict.get(failure)
        min_V = min_V_by_failure_mode.get(failure)

        num_points = 1000
        ls = torch.linspace(t_star, t, num_points).to(device)
        vals = integrand(all_sensor_readings, approx_cov_results, unit_manufac, sensors, ls, b, psi, beta, gamma, unit,
                         failure, min_V, data, hyperparameters,
                         lambda_hyperparameter)
        integral_approx = torch.trapz(vals, ls)
        s[failure] = p * torch.exp(-integral_approx)

    s_sum = sum(s.values())
    return s_sum


# def St_cond_EST(min_V_by_failure_mode, unit_f_probabilities, failure_modes, all_sensor_readings, approx_cov_results,
#                 unit_manufac, sensors, unit, t_star, t, b_dict, psi_dict, beta_dict, gamma_dict, data, hyperparameters,
#                 lambda_hyperparameter, unit_f, loaded_hyperparameters_fm,
#                 loaded_lambda_hyp_fm):
#     s = {}
#     for failure in failure_modes:
#         p = unit_f_probabilities.get(unit).get(failure)
#         b = b_dict.get(failure)
#         psi = psi_dict.get(failure)
#         # beta = beta_dict.get(unit_failure)
#         beta_values = []
#         for sensor in sensors:
#             beta_value = beta_dict.get((sensor, failure))
#             if beta_value is not None:
#                 beta_values.append(beta_value)
#             else:
#                 raise ValueError(f"Beta value for sensor {sensor} and failure mode {failure} not found.")
#         beta = torch.tensor(beta_values, device=device)
#         gamma = gamma_dict.get(failure)
#         min_V = min_V_by_failure_mode.get(failure)
#
#         num_points = 1000
#         ls = torch.linspace(t_star, t, num_points).to(device)
#
#         vals = (all_sensor_readings, approx_cov_results, unit_manufac, sensors, ls, b, psi, beta, gamma, unit,
#                 failure, min_V, data, hyperparameters, lambda_hyperparameter, unit_f, loaded_hyperparameters_fm,
#                 loaded_lambda_hyp_fm)
#         integral_approx = torch.trapz(vals, ls)
#         s[failure] = p * torch.exp(-integral_approx)
#
#     s_sum = sum(s.values())
#     return s_sum


def St_cond_EST_actual(unit_failure_mode, min_V_by_failure_mode, failure_modes, all_sensor_readings, approx_cov_results,
                       unit_manufac, sensors, unit, t_star, t, b_dict, psi_dict, beta_dict, gamma_dict, data,
                       hyperparameters,
                       lambda_hyperparameter):
    failure = unit_failure_mode.get(unit)
    b = b_dict.get(failure)
    psi = psi_dict.get(failure)
    # beta = beta_dict.get(unit_failure)
    beta_values = []
    for sensor in sensors:
        beta_value = beta_dict.get((sensor, failure))
        if beta_value is not None:
            beta_values.append(beta_value)
        else:
            raise ValueError(f"Beta value for sensor {sensor} and failure mode {failure} not found.")
    beta = torch.tensor(beta_values, device=device)
    gamma = gamma_dict.get(failure)
    min_V = min_V_by_failure_mode.get(failure)

    num_points = 1000
    ls = torch.linspace(t_star, t, num_points).to(device)
    vals = integrand(all_sensor_readings, approx_cov_results, unit_manufac, sensors, ls, b, psi, beta, gamma, unit,
                     failure, min_V, data, hyperparameters,
                     lambda_hyperparameter)
    integral_approx = torch.trapz(vals, ls)
    return torch.exp(-integral_approx)


def integrate_St_cond_EST(min_V_by_failure_mode, unit_f_probabilities, failure_modes, all_sensor_readings,
                          approx_cov_results, unit_manufacturer, sensors, unit, t_star, t_end, b_dict, psi_dict,
                          beta_dict,
                          gamma_dict, data, hyperparameters, lambda_hyperparameter, num_points=100):
    ts = torch.linspace(t_star, t_end, num_points).to(device)
    integrand_vals = torch.zeros(num_points).to(device)
    for i, t in enumerate(ts):
        integrand_vals[i] = St_cond_EST(min_V_by_failure_mode, unit_f_probabilities, failure_modes, all_sensor_readings,
                                        approx_cov_results,
                                        unit_manufacturer, sensors, unit, t_star, t, b_dict, psi_dict, beta_dict,
                                        gamma_dict,
                                        data,
                                        hyperparameters,
                                        lambda_hyperparameter)
    integral_value = torch.trapz(integrand_vals, ts)
    return integral_value


# def integrate_St_cond_EST(min_V_by_failure_mode, unit_f_probabilities, failure_modes, all_sensor_readings,
#                           approx_cov_results, unit_manufacturer, sensors, unit, t_star, t_end, b_dict, psi_dict,
#                           beta_dict,
#                           gamma_dict, data, hyperparameters, lambda_hyperparameter, unit_f, loaded_hyperparameters_fm, loaded_lambda_hyp_fm, num_points=100):
#     ts = torch.linspace(t_star, t_end, num_points).to(device)
#     integrand_vals = torch.zeros(num_points).to(device)
#     for i, t in enumerate(ts):
#         integrand_vals[i] = St_cond_EST(min_V_by_failure_mode, unit_f_probabilities, failure_modes, all_sensor_readings,
#                                         approx_cov_results,
#                                         unit_manufacturer, sensors, unit, t_star, t, b_dict, psi_dict, beta_dict,
#                                         gamma_dict,
#                                         data,
#                                         hyperparameters,
#                                         lambda_hyperparameter, unit_f, loaded_hyperparameters_fm, loaded_lambda_hyp_fm)
#     integral_value = torch.trapz(integrand_vals, ts)
#     return integral_value


def integrate_St_cond_EST_actual(unit_failure_mode, min_V_by_failure_mode, failure_modes, all_sensor_readings,
                                 approx_cov_results, unit_manufacturer, sensors, unit, t_star, t_end, b_dict, psi_dict,
                                 beta_dict,
                                 gamma_dict, data, hyperparameters, lambda_hyperparameter, num_points=100):
    ts = torch.linspace(t_star, t_end, num_points).to(device)
    integrand_vals = torch.zeros(num_points).to(device)
    for i, t in enumerate(ts):
        integrand_vals[i] = St_cond_EST_actual(unit_failure_mode, min_V_by_failure_mode, failure_modes,
                                               all_sensor_readings,
                                               approx_cov_results,
                                               unit_manufacturer, sensors, unit, t_star, t, b_dict, psi_dict, beta_dict,
                                               gamma_dict,
                                               data,
                                               hyperparameters,
                                               lambda_hyperparameter)
    integral_value = torch.trapz(integrand_vals, ts)
    return integral_value


########################################################################################################################


def probability_of_failure_hist_unit(slice_value, data_dicts, all_sensor_readings, loaded_hyperparameters,
                                     loaded_lambda_hyp, loaded_hyperparameters_fm, loaded_lambda_hyp_fm,
                                     approx_cov_results, unit, actual_failure, pi_hat, sensor, unit_f):
    data_dict = data_dicts.get((sensor, actual_failure))
    historical_inputs = torch.tensor(data_dict[(unit, sensor, actual_failure)]['time_points'][:slice_value],
                                     device=device)

    historical_outputs = torch.tensor(
        data_dict[(unit, sensor, actual_failure)]['sensor_readings'][:slice_value],
        dtype=torch.float32,
        device=device
    ).unsqueeze(-1)

    if (unit, 1) in unit_f:
        means_1, variance_1 = get_cmgp_predictions(
            all_sensor_readings, historical_inputs, inducing_points_num, unit,
            sensor, 1, data_dicts, loaded_hyperparameters,
            loaded_lambda_hyp, approx_cov_results, preferred_device=device
        )
    else:
        means_1, variance_1 = get_cmgp_predictions_fm(
            all_sensor_readings, historical_inputs, inducing_points_num, unit,
            sensor, 1, data_dicts, loaded_hyperparameters_fm,
            loaded_lambda_hyp_fm, approx_cov_results, preferred_device=device
        )

    if (unit, 2) in unit_f:
        means_2, variance_2 = get_cmgp_predictions(
            all_sensor_readings, historical_inputs, inducing_points_num, unit,
            sensor, 2, data_dicts, loaded_hyperparameters,
            loaded_lambda_hyp, approx_cov_results, preferred_device=device
        )

    else:
        means_2, variance_2 = get_cmgp_predictions_fm(
            all_sensor_readings, historical_inputs, inducing_points_num, unit,
            sensor, 2, data_dicts, loaded_hyperparameters_fm,
            loaded_lambda_hyp_fm, approx_cov_results, preferred_device=device
        )

    if (unit, 3) in unit_f:
        means_3, variance_3 = get_cmgp_predictions(
            all_sensor_readings, historical_inputs, inducing_points_num, unit,
            sensor, 3, data_dicts, loaded_hyperparameters,
            loaded_lambda_hyp, approx_cov_results, preferred_device=device
        )

    else:
        means_3, variance_3 = get_cmgp_predictions_fm(
            all_sensor_readings, historical_inputs, inducing_points_num, unit,
            sensor, 3, data_dicts, loaded_hyperparameters_fm,
            loaded_lambda_hyp_fm, approx_cov_results, preferred_device=device
        )
    prob1, log_like1 = likelihood(historical_outputs, means_1, variance_1)
    prob2, log_like2 = likelihood(historical_outputs, means_2, variance_2)
    prob3, log_like3 = likelihood(historical_outputs, means_3, variance_3)

    max_log_likelihood = max(log_like1, log_like2, log_like3).item()

    # Check if the maximum log-likelihood is greater than 88 and adjust if necessary
    if max_log_likelihood > 88:
        subtract_value = max_log_likelihood - 88

        # Subtract the necessary value from all log-likelihoods
        log_like1 -= subtract_value
        log_like2 -= subtract_value
        log_like3 -= subtract_value

        # Recalculate the probabilities after adjustment
        prob1 = torch.exp(log_like1)
        prob2 = torch.exp(log_like2)
        prob3 = torch.exp(log_like3)

    # print(f'unit:{unit} ==> Re_log_likelihood_1: {log_like1.item()}, Re_log_likelihood_2: {log_like2.item()}, '
    #       f'Re_log_likelihood_3: {log_like3.item()}')
    # print(
    #     f'unit:{unit} ==> Re_likelihood_1: {prob1.item()}, Re_likelihood_2: {prob2.item()}, Re_likelihood_3: {prob3.item()}')

    pi1 = pi_hat.get(1)
    pi2 = pi_hat.get(2)
    pi3 = pi_hat.get(3)
    n = len(historical_inputs)

    denominator = pi1 * prob1 + pi2 * prob2 + pi3 * prob3

    prob_one = (pi1 * prob1) / denominator
    prob_two = (pi2 * prob2) / denominator
    prob_three = (pi3 * prob3) / denominator

    # Check for NaN and replace with 1 if necessary
    prob_one = torch.where(torch.isnan(prob_one), torch.tensor(1.0), prob_one)
    prob_two = torch.where(torch.isnan(prob_two), torch.tensor(1.0), prob_two)
    prob_three = torch.where(torch.isnan(prob_three), torch.tensor(1.0), prob_three)

    return prob_one, prob_two, prob_three


def likelihood_of_failure_all_test_ns(data_dict_test, data_dicts, all_sensor_readings, loaded_hyperparameters,
                                      loaded_lambda_hyp,
                                      approx_cov_results, unit, actual_failure, pi_hat, sensor):
    data_dict = data_dict_test.get((sensor, actual_failure))
    historical_inputs = torch.tensor(data_dict[(unit, sensor, actual_failure)]['time_points'],
                                     device=device)

    historical_outputs = torch.tensor(data_dict[(unit, sensor, actual_failure)]['sensor_readings'],
                                      dtype=torch.float32,
                                      device=device).unsqueeze(-1)

    means_1, variance_1 = get_cmgp_predictions(all_sensor_readings, historical_inputs, inducing_points_num, unit,
                                               [sensor],
                                               1,
                                               data_dicts, loaded_hyperparameters,
                                               loaded_lambda_hyp, approx_cov_results, preferred_device=device)

    means_2, variance_2 = get_cmgp_predictions(all_sensor_readings, historical_inputs, inducing_points_num, unit,
                                               [sensor],
                                               2,
                                               data_dicts, loaded_hyperparameters,
                                               loaded_lambda_hyp, approx_cov_results, preferred_device=device)

    prob1, log_like1 = likelihood(historical_outputs.squeeze().unsqueeze(-1), means_1.squeeze().unsqueeze(-1),
                                  variance_1.squeeze())
    prob2, log_like2 = likelihood(historical_outputs.squeeze().unsqueeze(-1), means_2.squeeze().unsqueeze(-1),
                                  variance_2.squeeze())

    # print(f'unit:{unit} ==> log_likelihood_1: {log_like1.item()}, log_likelihood_2: {log_like2.item()}, '
    #       f'log_likelihood_3: {log_like3.item()}')
    # print(f'unit:{unit} ==> likelihood_1: {prob1.item()}, likelihood_2: {prob2.item()}, likelihood_3: {prob3.item()}')

    # Check if the maximum log-likelihood is greater than 88 and adjust if necessary
    # max_log_likelihood = max(log_like1, log_like2).item()
    #
    # # Check if the maximum log-likelihood is greater than 88 and adjust if necessary
    # if max_log_likelihood > 88:
    #     subtract_value = max_log_likelihood - 88
    #
    #     # Subtract the necessary value from all log-likelihoods
    #     log_like1 -= subtract_value
    #     log_like2 -= subtract_value
    #
    #     # Recalculate the probabilities after adjustment
    #     prob1 = torch.exp(log_like1)
    #     prob2 = torch.exp(log_like2)

    max_log_likelihood = max(log_like1, log_like2).item()
    min_log_likelihood = min(log_like1, log_like2).item()
    #
    # # Check if the maximum log-likelihood is greater than 88 and adjust if necessary
    # if max_log_likelihood > 88:
    #     subtract_value = max_log_likelihood - 88
    #
    #     # Subtract the necessary value from all log-likelihoods
    #     log_like1 -= subtract_value
    #     log_like2 -= subtract_value
    #
    # # Check if the minimum log-likelihood is less than -88 and adjust if necessary
    # if min_log_likelihood < -88:
    #     add_value = -88 - min_log_likelihood
    #
    #     # Add the necessary value to all log-likelihoods
    #     log_like1 += add_value
    #     log_like2 += add_value

    #########
    if max_log_likelihood > 88 or min_log_likelihood < -88:
        # Calculate the midpoint of the current log-likelihood range
        mid_log_likelihood = (max_log_likelihood + min_log_likelihood) / 2

        # Calculate the adjustment to bring the midpoint to 0, centered within [-88, 88]
        shift_value = mid_log_likelihood

        # Subtract the shift value from all log-likelihoods to center them
        log_like1 -= shift_value
        log_like2 -= shift_value

        # After centering, ensure they are within the [-88, 88] range
        # If still out of range, scale them uniformly to fit in [-88, 88]
        max_log_likelihood = max(log_like1, log_like2)
        min_log_likelihood = min(log_like1, log_like2)

        # Check if scaling is needed
        if max_log_likelihood > 88 or min_log_likelihood < -88:
            # Calculate scaling factor to bring them within the range
            scaling_factor = 88 / max(abs(max_log_likelihood), abs(min_log_likelihood))

            # Scale log-likelihoods uniformly
            log_like1 *= scaling_factor
            log_like2 *= scaling_factor

    #####################
    # Recalculate the probabilities after adjustment
    prob1 = torch.exp(log_like1)
    prob2 = torch.exp(log_like2)

    c = 1 / 1
    return c * prob1, c * prob2


########################################################################################################################
########################################################################################################################
def h0_2(l, b, rho, min_V):
    return torch.where(l >= min_V, torch.exp(b + rho * (l - min_V)), torch.tensor(0.0))


def integrand_2(all_sensor_readings, approx_cov_results, unit_manufac, sensors, l, mu_b_hat, sigma_b_hat, alpha_rho_hat,
                beta_rho_hat, beta, gamma, unit,
                unit_failure, min_V, data, hyperparameters, lambda_hyperparameter):
    b = torch.normal(mu_b_hat, sigma_b_hat)
    gamma_dist = torch.distributions.gamma.Gamma(alpha_rho_hat, beta_rho_hat)
    rho = gamma_dist.sample()

    # b = mu_b_hat
    # rho = alpha_rho_hat / beta_rho_hat

    h0_value = h0_2(l, b, rho, min_V)

    means, cov_matrix = get_cmgp_predictions(all_sensor_readings, l, inducing_points_num, unit,
                                             sensors,
                                             unit_failure,
                                             data, hyperparameters,
                                             lambda_hyperparameter, approx_cov_results, preferred_device=device)

    means = means.squeeze()
    # variances = torch.diagonal(variances)
    variances = torch.diagonal(cov_matrix, dim1=-2, dim2=-1).squeeze(0)

    samples = torch.normal(means, torch.sqrt(variances))

    if beta.shape == torch.Size([1]):
        exp_component_first = beta * samples

    else:
        exp_component_first = beta @ samples
    # exp_component_third = gamma * unit_manufac.get(unit)
    exp_component_third = torch.tensor(0.0, device=device)
    exp_component = exp_component_first + exp_component_third

    return h0_value * torch.exp(exp_component)


def St_cond_EST_2(min_V_by_failure_mode, unit_f_probabilities, failure_modes, all_sensor_readings, approx_cov_results,
                  unit_manufac, sensors, unit, t_star, t, mu_b_hat_dict, sigma_b_hat_dict, alpha_rho_hat_dict,
                  beta_rho_hat_dict, beta_dict, gamma_dict, data,
                  hyperparameters,
                  lambda_hyperparameter, num_samples=20):
    s_distribution = []

    for _ in range(num_samples):
        s = {}
        for failure in failure_modes:
            p = unit_f_probabilities.get(unit).get(failure)
            mu_b_hat = mu_b_hat_dict.get(failure)
            sigma_b_hat = sigma_b_hat_dict.get(failure)
            alpha_rho_hat = alpha_rho_hat_dict.get(failure)
            beta_rho_hat = beta_rho_hat_dict.get(failure)

            # beta = beta_dict.get(unit_failure)
            beta_values = []
            for sensor in sensors:
                beta_value = beta_dict.get((sensor, failure))
                if beta_value is not None:
                    beta_values.append(beta_value)
                else:
                    raise ValueError(f"Beta value for sensor {sensor} and failure mode {failure} not found.")
            beta = torch.tensor(beta_values, device=device)
            gamma = gamma_dict.get(failure)
            min_V = min_V_by_failure_mode.get(failure)

            num_points = 1000
            ls = torch.linspace(t_star, t, num_points).to(device)
            vals = integrand_2(all_sensor_readings, approx_cov_results, unit_manufac, sensors, ls, mu_b_hat,
                               sigma_b_hat,
                               alpha_rho_hat,
                               beta_rho_hat, beta, gamma,
                               unit,
                               failure, min_V, data, hyperparameters,
                               lambda_hyperparameter)
            integral_approx = torch.trapz(vals, ls)
            s[failure] = p * torch.exp(-integral_approx)

        s_sum = sum(s.values())
        s_distribution.append(s_sum)

    return torch.tensor(s_distribution)
