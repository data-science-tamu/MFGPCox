import torch
from torch import lgamma

########################################################################################################################

# specifying the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

########################################################################################################################


# kernel for the latent function
def k_u_u(t, t_prime, length_scale):
    return (1 / (torch.sqrt(2 * torch.pi * length_scale ** 2)) * torch.exp(
        -0.5 * (t - t_prime) ** 2 / length_scale ** 2))
    # return torch.exp(-0.5 * ((t - t_prime) ** 2) / length_scale ** 2)


# kernel for covariance between two functions
def k_f_f(t, t_prime, alpha, alpha_prime, xi, xi_prime, lambda_hyp):
    eta_squared = xi ** 2 + xi_prime ** 2 + lambda_hyp ** 2
    return (alpha * alpha_prime / (torch.sqrt(2 * torch.pi * eta_squared))
            * torch.exp(-0.5 * (t - t_prime) ** 2 / eta_squared))
    # return alpha * alpha_prime * torch.exp(-0.5 * (t - t_prime) ** 2 / eta_squared)


# kernel for covariance between a function and the latent function
def k_f_u(t, w, alpha, xi, lambda_hyp):
    eta_squared = xi ** 2 + lambda_hyp ** 2
    return alpha / (torch.sqrt(2 * torch.pi * eta_squared)) * torch.exp(-0.5 * (t - w) ** 2 / eta_squared)
    # return alpha * torch.exp(-0.5 * (t - w) ** 2 / eta_squared)


def k_f_f_2(t, t_prime, mu_alpha_hat, sigma_alpha_hat, alpha_theta1_hat, beta_theta1_hat):
    first = (mu_alpha_hat ** 2 + sigma_alpha_hat ** 2) / (torch.sqrt(torch.tensor(2.0, device=device) * torch.pi))
    second = (beta_theta1_hat ** alpha_theta1_hat * torch.exp(lgamma(alpha_theta1_hat + 1 / 2))) / torch.exp(
        lgamma(alpha_theta1_hat))
    third = 1 / ((((t - t_prime) ** 2) / 2) + beta_theta1_hat) ** (alpha_theta1_hat + 1 / 2)

    return first * second * third


# kernel for covariance between a function and the latent function
def k_f_u_2(t, w, mu_alpha_hat, alpha_theta2_hat, beta_theta2_hat):
    first = mu_alpha_hat / (torch.sqrt(torch.tensor(2.0, device=device) * torch.pi))
    second = (beta_theta2_hat ** alpha_theta2_hat * torch.exp(lgamma(alpha_theta2_hat + 1 / 2))) / torch.exp(
        lgamma(alpha_theta2_hat))
    third = 1 / (((((t - w) ** 2) / 2) + beta_theta2_hat) ** (alpha_theta2_hat + 1 / 2))

    return first * second * third


def k_f_f_2_pred(t, t_prime, alpha, theta_1):
    return ((alpha ** 2) * torch.sqrt(theta_1) / (torch.sqrt(torch.tensor(2.0, device=device) * torch.pi)) *
            torch.exp(-0.5 * (t - t_prime) ** 2 * theta_1))


# kernel for covariance between a function and the latent function
def k_f_u_2_pred(t, w, alpha, theta_2):
    return ((alpha ** 2) * torch.sqrt(theta_2) / (torch.sqrt(torch.tensor(2.0, device=device) * torch.pi)) *
            torch.exp(-0.5 * (t - w) ** 2 * theta_2))


########################################################################################################################

# Calculate the GP sparse covariance matrix for a specific unit, sensor, and failure mode
def get_approximated_covariance_matrix(data, hyperparameters, lambda_hyperparameter,
                                       number_of_inducing_points,
                                       preferred_device=device):
    # defining empty lists to store D and noise matrices for each function, so we use these lists to build the
    # block diagonal matrix D
    d_ijk_matrices = []
    noise_ijk_matrices = []
    k_f_u_matrices = []

    inducing_points = get_inducing_points(data, number_of_inducing_points)

    # inducing_points = inducing_points.double()

    # calculating and inverting the covariance matrix of the latent function (evaluated at inducing points)
    k_u_u_matrix = k_u_u(inducing_points, inducing_points.T, lambda_hyperparameter)
    # k_u_u_inv = invert_matrix(k_u_u_matrix)

    # calculating D_ijk matrices which is specific to each function we have
    for (i, j, k), time_series in data.items():
        input_points = torch.tensor(time_series['time_points'], device=preferred_device).unsqueeze(-1)
        selected_function_hyperparameters = hyperparameters[(j, k)]
        alpha = selected_function_hyperparameters['alpha'].clone()
        xi = selected_function_hyperparameters['xi'].clone()
        sigma = selected_function_hyperparameters['sigma'].clone()

        k_f_f_matrix = k_f_f(input_points, input_points.T, alpha, alpha, xi, xi, lambda_hyperparameter)
        k_f_u_matrix = k_f_u(input_points, inducing_points.T, alpha, xi, lambda_hyperparameter)

        # d_ijk = k_f_f_matrix - k_f_u_matrix @ k_u_u_inv @ k_f_u_matrix.T

        # added this next two lines for numerical study
        # k_u_u_matrix = k_u_u_matrix.float()
        # k_f_u_matrix = k_f_u_matrix.float()

        d_ijk = k_f_f_matrix - k_f_u_matrix @ torch.linalg.solve(k_u_u_matrix, k_f_u_matrix.T)

        d_ijk_matrices.append(d_ijk)
        k_f_u_matrices.append(k_f_u_matrix)
        identity = torch.eye(d_ijk.shape[0], device=device)
        noise_ijk_matrices.append((sigma * identity))

    # building block-diagonal D and noise matrices
    d = torch.block_diag(*d_ijk_matrices)
    noise = torch.block_diag(*noise_ijk_matrices)

    # building the stacked vectors of K_fu and K_uf matrices
    k_f_u_stacked = torch.cat(k_f_u_matrices, dim=0)
    k_u_f_stacked = k_f_u_stacked.transpose(0, 1)

    # approximated_covariance_matrix = d + k_f_u_stacked @ k_u_u_inv @ k_u_f_stacked + noise
    approximated_covariance_matrix = d + k_f_u_stacked @ torch.linalg.solve(k_u_u_matrix, k_u_f_stacked) + noise

    # followings are for prediction and visualization purpose

    # calculating A and A inverse matrix for prediction
    d_plus_noise_matrix = d + noise
    d_plus_noise_matrix_inv = torch.linalg.inv(d_plus_noise_matrix)
    # a = k_u_u_matrix + k_u_f_stacked @ d_plus_noise_matrix_inv @ k_f_u_stacked
    a = k_u_u_matrix + k_u_f_stacked @ torch.linalg.solve(d_plus_noise_matrix, k_f_u_stacked)

    a_inverse_matrix = torch.linalg.inv(a)

    # u_mean_m = k_u_u_matrix @ a_inverse_matrix @ k_u_f_stacked @ d_plus_noise_matrix_inv
    u_mean_m = k_u_u_matrix @ torch.linalg.solve(a, k_u_f_stacked)

    torch.linalg.solve(a, k_u_f_stacked)

    return (approximated_covariance_matrix, a, d_plus_noise_matrix, d_plus_noise_matrix_inv,
            k_u_f_stacked, u_mean_m)


def get_approximated_covariance_matrix_2(data, hyperparameters, lambda_hyperparameter,
                                         number_of_inducing_points,
                                         preferred_device=device):
    # defining empty lists to store D and noise matrices for each function, so we use these lists to build the
    # block diagonal matrix D
    d_ijk_matrices = []
    noise_ijk_matrices = []
    k_f_u_matrices = []

    inducing_points = get_inducing_points(data, number_of_inducing_points)

    # inducing_points = inducing_points.double()

    # calculating and inverting the covariance matrix of the latent function (evaluated at inducing points)
    k_u_u_matrix = k_u_u(inducing_points, inducing_points.T, lambda_hyperparameter)
    # k_u_u_inv = invert_matrix(k_u_u_matrix)

    # calculating D_ijk matrices which is specific to each function we have
    for (i, j, k), time_series in data.items():
        input_points = torch.tensor(time_series['time_points'], device=preferred_device).unsqueeze(-1)
        selected_function_hyperparameters = hyperparameters[(j, k)]
        mu_alpha_hat = selected_function_hyperparameters['mu_alpha_hat'].clone()
        sigma_alpha_hat = selected_function_hyperparameters['sigma_alpha_hat'].clone()
        alpha_theta1_hat = selected_function_hyperparameters['alpha_theta1_hat'].clone()
        beta_theta1_hat = selected_function_hyperparameters['beta_theta1_hat'].clone()
        alpha_theta2_hat = selected_function_hyperparameters['alpha_theta2_hat'].clone()
        beta_theta2_hat = selected_function_hyperparameters['beta_theta2_hat'].clone()
        sigma = selected_function_hyperparameters['sigma'].clone()

        k_f_f_matrix = k_f_f_2(input_points, input_points.T, mu_alpha_hat, sigma_alpha_hat, alpha_theta1_hat,
                               beta_theta1_hat)
        k_f_u_matrix = k_f_u_2(input_points, inducing_points.T, mu_alpha_hat, alpha_theta2_hat, beta_theta2_hat)

        # d_ijk = k_f_f_matrix - k_f_u_matrix @ k_u_u_inv @ k_f_u_matrix.T

        # added this next two lines for numerical study
        # k_u_u_matrix = k_u_u_matrix.float()
        # k_f_u_matrix = k_f_u_matrix.float()

        d_ijk = k_f_f_matrix - k_f_u_matrix @ torch.linalg.solve(k_u_u_matrix, k_f_u_matrix.T)

        d_ijk_matrices.append(d_ijk)
        k_f_u_matrices.append(k_f_u_matrix)
        identity = torch.eye(d_ijk.shape[0], device=device)
        noise_ijk_matrices.append((sigma * identity))

    # building block-diagonal D and noise matrices
    d = torch.block_diag(*d_ijk_matrices)
    noise = torch.block_diag(*noise_ijk_matrices)

    # building the stacked vectors of K_fu and K_uf matrices
    k_f_u_stacked = torch.cat(k_f_u_matrices, dim=0)
    k_u_f_stacked = k_f_u_stacked.transpose(0, 1)

    # approximated_covariance_matrix = d + k_f_u_stacked @ k_u_u_inv @ k_u_f_stacked + noise
    approximated_covariance_matrix = d + k_f_u_stacked @ torch.linalg.solve(k_u_u_matrix, k_u_f_stacked) + noise

    # followings are for prediction and visualization purpose

    # calculating A and A inverse matrix for prediction
    d_plus_noise_matrix = d + noise
    d_plus_noise_matrix_inv = torch.linalg.inv(d_plus_noise_matrix)
    # a = k_u_u_matrix + k_u_f_stacked @ d_plus_noise_matrix_inv @ k_f_u_stacked
    a = k_u_u_matrix + k_u_f_stacked @ torch.linalg.solve(d_plus_noise_matrix, k_f_u_stacked)

    a_inverse_matrix = torch.linalg.inv(a)

    # u_mean_m = k_u_u_matrix @ a_inverse_matrix @ k_u_f_stacked @ d_plus_noise_matrix_inv
    u_mean_m = k_u_u_matrix @ torch.linalg.solve(a, k_u_f_stacked)

    torch.linalg.solve(a, k_u_f_stacked)

    return (approximated_covariance_matrix, a, d_plus_noise_matrix, d_plus_noise_matrix_inv,
            k_u_f_stacked, u_mean_m)


########################################################################################################################

# useful functions

# function to get inducing points
def get_inducing_points(data, number_of_inducing_points, preferred_device=device):
    """specifying inducing points by considering the length of longest time series and specifying inducing points as
    equally spaced points in the selected range"""

    longest_time_series = data[max(data, key=lambda k: len(data[k]['time_points']))]

    length = longest_time_series['time_points']
    inducing_points = torch.linspace(length.min() - 30, torch.tensor(length.max() + 30, device=device),
                                     steps=number_of_inducing_points,
                                     device=preferred_device).unsqueeze(-1)

    return inducing_points


# function to do matrix inversion while adding some small values to the diagonal (jitter)
def invert_matrix(matrix):
    # small values added to the diagonal to avoid singularity in inversion
    epsilon = torch.tensor([1e-10]).to(device)
    identity_matrix = torch.eye(matrix.size(0), device=device)

    matrix_adjusted = matrix + epsilon * identity_matrix
    matrix_inv = torch.linalg.inv(matrix_adjusted)

    # # inverting using Cholesky decomposition
    # L = torch.linalg.cholesky(K_u_u_matrix_adjusted)
    # K_u_u_inv = torch.cholesky_inverse(L)

    return matrix_inv


########################################################################################################################
def neg_log_likelihood(all_sensor_readings, data, hyperparameters, lambda_hyp, number_of_inducing_points,
                       preferred_device=device):
    y = torch.tensor(all_sensor_readings, dtype=torch.float32, device=preferred_device).unsqueeze(-1)
    K, _, _, _, _, _ = get_approximated_covariance_matrix(data, hyperparameters, lambda_hyp,
                                                          number_of_inducing_points)
    _, log_det = torch.linalg.slogdet(K)

    log_likelihood = -0.5 * y.transpose(0, 1) @ torch.linalg.solve(K, y) - 0.5 * log_det

    # log_likelihood = (-0.5 * y.transpose(0, 1) @ torch.linalg.solve(K, y) - 0.5 * log_det
    #                   - y.size(0) / 2 * torch.log(torch.tensor(2 * torch.pi, device=device)))

    return - log_likelihood


def neg_log_likelihood_2(all_sensor_readings, data, hyperparameters, lambda_hyp, number_of_inducing_points,
                         preferred_device=device):
    y = torch.tensor(all_sensor_readings, dtype=torch.float32, device=preferred_device).unsqueeze(-1)
    K, _, _, _, _, _ = get_approximated_covariance_matrix_2(data, hyperparameters, lambda_hyp,
                                                            number_of_inducing_points)
    _, log_det = torch.linalg.slogdet(K)

    log_likelihood = -0.5 * y.transpose(0, 1) @ torch.linalg.solve(K, y) - 0.5 * log_det

    # log_likelihood = (-0.5 * y.transpose(0, 1) @ torch.linalg.solve(K, y) - 0.5 * log_det
    #                   - y.size(0) / 2 * torch.log(torch.tensor(2 * torch.pi, device=device)))

    return - log_likelihood


########################################################################################################################

def get_cmgp_predictions(all_sensor_readings, prediction_input, number_of_inducing_points, unit, sensor,
                         prediction_failure_mode,
                         data_dicts, hyperparameters, lambda_hyperparameter_dict, approx_cov, preferred_device=device):
    data = data_dicts.get(prediction_failure_mode)
    inducing_points = get_inducing_points(data, number_of_inducing_points)

    lambda_hyperparameter = lambda_hyperparameter_dict.get(prediction_failure_mode)
    k_u_u_matrix = k_u_u(inducing_points, inducing_points.T, lambda_hyperparameter)

    historical_outputs = all_sensor_readings[prediction_failure_mode]
    historical_outputs = torch.tensor(historical_outputs, dtype=torch.float32, device=preferred_device).unsqueeze(-1)

    t_star = prediction_input.clone().detach().unsqueeze(-1).to(preferred_device)

    params = hyperparameters[(sensor, prediction_failure_mode)]
    alpha = params['alpha'].clone()
    xi = params['xi'].clone()
    sigma = params['sigma'].clone()

    k_f_star_u_matrix = k_f_u(t_star, inducing_points.T, alpha, xi, lambda_hyperparameter)
    k_u_f_star_matrix = k_f_star_u_matrix.transpose(0, 1)

    k_f_star_f_star_matrix = k_f_f(t_star, t_star.T, alpha, alpha, xi, xi, lambda_hyperparameter)

    k_u_f_stacked = approx_cov[prediction_failure_mode]['k_u_f_stacked']
    d_plus_noise_matrix = approx_cov[prediction_failure_mode]['d_plus_noise_matrix']
    a_matrix = approx_cov[prediction_failure_mode]['a']

    mean_prediction = (k_f_star_u_matrix @ torch.linalg.solve(a_matrix, k_u_f_stacked) @
                       torch.linalg.solve(d_plus_noise_matrix, historical_outputs))

    var_pred = (k_f_star_f_star_matrix - k_f_star_u_matrix @ torch.linalg.solve(k_u_u_matrix,
                                                                                k_u_f_star_matrix) + k_f_star_u_matrix @
                torch.linalg.solve(a_matrix, k_u_f_star_matrix))  # + sigma * torch.eye(k_f_star_f_star_matrix.shape[0],
    #           device=preferred_device))

    return mean_prediction, var_pred

def gamma_mode(shape, rate):
    if shape <= 1:
        return torch.tensor(0.0)
    return (shape - 1) / rate


def get_cmgp_predictions_2(all_sensor_readings, prediction_input, number_of_inducing_points, unit, sensor,
                           prediction_failure_mode,
                           data_dicts, hyperparameters, lambda_hyperparameter_dict, approx_cov,
                           preferred_device=device):
    data = data_dicts.get(prediction_failure_mode)

    inducing_points = get_inducing_points(data, number_of_inducing_points)

    lambda_hyperparameter = lambda_hyperparameter_dict.get(prediction_failure_mode)
    k_u_u_matrix = k_u_u(inducing_points, inducing_points.T, lambda_hyperparameter)

    historical_outputs = all_sensor_readings[prediction_failure_mode]
    historical_outputs = torch.tensor(historical_outputs, dtype=torch.float32, device=preferred_device).unsqueeze(-1)

    t_star = prediction_input.clone().detach().unsqueeze(-1).to(preferred_device)

    params = hyperparameters[(sensor, prediction_failure_mode)]
    mu_alpha_hat = params['mu_alpha_hat'].clone()
    sigma_alpha_hat = params['sigma_alpha_hat'].clone()
    alpha_theta1_hat = params['alpha_theta1_hat'].clone()
    beta_theta1_hat = params['beta_theta1_hat'].clone()
    alpha_theta2_hat = params['alpha_theta2_hat'].clone()
    beta_theta2_hat = params['beta_theta2_hat'].clone()

    sigma = params['sigma'].clone()

    k_f_star_u_matrix = k_f_u_2_pred(t_star, inducing_points.T, mu_alpha_hat, gamma_mode(alpha_theta2_hat, beta_theta2_hat))
    # k_f_star_u_matrix = k_f_u_2_pred(t_star, inducing_points.T, mu_alpha_hat, alpha_theta2_hat / beta_theta2_hat)

    k_u_f_star_matrix = k_f_star_u_matrix.transpose(0, 1)

    k_f_star_f_star_matrix = k_f_f_2_pred(t_star, t_star.T, mu_alpha_hat, gamma_mode(alpha_theta1_hat,beta_theta1_hat))

    # k_f_star_f_star_matrix = k_f_f_2_pred(t_star, t_star.T, mu_alpha_hat, alpha_theta1_hat/beta_theta1_hat)


    k_u_f_stacked = approx_cov[prediction_failure_mode]['k_u_f_stacked']
    d_plus_noise_matrix = approx_cov[prediction_failure_mode]['d_plus_noise_matrix']
    a_matrix = approx_cov[prediction_failure_mode]['a']

    mean_prediction = (k_f_star_u_matrix @ torch.linalg.solve(a_matrix, k_u_f_stacked) @
                       torch.linalg.solve(d_plus_noise_matrix, historical_outputs))

    var_pred = (k_f_star_f_star_matrix - k_f_star_u_matrix @ torch.linalg.solve(k_u_u_matrix,
                                                                                k_u_f_star_matrix) + k_f_star_u_matrix @
                torch.linalg.solve(a_matrix, k_u_f_star_matrix))  # + sigma * torch.eye(k_f_star_f_star_matrix.shape[0],
    #           device=preferred_device))

    return mean_prediction, var_pred
