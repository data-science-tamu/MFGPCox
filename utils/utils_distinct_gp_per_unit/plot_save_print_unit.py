from utils.utils_distinct_gp_per_unit.data_processing_unit import reconstruct_hyperparameters, \
    reconstruct_hyperparameters_elbo, reconstruct_hyperparameters_test, reconstruct_hyperparameters_elbo_2, \
    reconstruct_hyperparameters_test_2
import time
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
import logging


########################################################################################################################

def plot_covariance_matrix(cov_matrix, cov_matrix_name):
    fig, ax = plt.subplots()
    # fig=plt.figure(figsize=(100,10))
    cax = ax.matshow(cov_matrix.cpu(), cmap='jet',
                     aspect='auto')
    fig.colorbar(cax)

    plt.xlabel('input points')
    plt.ylabel('input points')
    plt.title(cov_matrix_name)
    plt.show()


########################################################################################################################

def format_time(seconds):
    """Format time based on the duration to display in days, hours, minutes, or seconds."""
    if seconds >= 86400:  # More than a day
        days = seconds // 86400
        hours = (seconds % 86400) // 3600
        return f"{int(days)}d {int(hours)}h"
    elif seconds >= 3600:  # Between an hour and a day
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{int(hours)}h {int(minutes)}m"
    elif seconds >= 60:  # Between a minute and an hour
        minutes = seconds // 60
        sec = seconds % 60
        return f"{int(minutes)}m {int(sec)}s"
    else:  # Less than a minute
        return f"{int(seconds)}s"


def red_text(text):
    """Return the text in red color."""
    return f"\033[91m{text}\033[0m"


########################################################################################################################


# Print and save for gp hyperparameters
def save_and_print_gp_hyperparams(iteration, num_iterations, losses, flat_parameters_initial, initial_hyperparameters,
                                  main_folder_path, file_name, save_interval=10000,
                                  print_interval=1, header_written=False,
                                  failure_modes=None, start_time=None):
    # Only execute the following block at the specified print interval
    if (iteration + 1) % print_interval == 0:
        # Reconstruct hyperparameters and lambda_hyp for each failure mode
        optimized_hyperparameters, optimized_lambda_hyp = reconstruct_hyperparameters(
            flat_parameters_initial.detach(), initial_hyperparameters, failure_modes)

        # Prepare data for saving
        total_loss = losses[-1][0]
        data_to_save = {
            'Iteration': iteration + 1,
            'Total_Loss': total_loss,
        }

        # Dynamically add losses for each failure mode
        fm_losses = {}
        for i, fm in enumerate(failure_modes):
            fm_losses[fm] = losses[-1][i + 1]
            data_to_save[f"fm{fm}_Loss"] = fm_losses[fm]

        # Include lambda_hyp values for each failure mode in the data to save
        for fm, lambda_val in optimized_lambda_hyp.items():
            data_to_save[f"Lambda_hyp_{fm}"] = lambda_val.item()

        # Include the hyperparameters in the data to save
        for key, params in optimized_hyperparameters.items():
            for param_name, param_value in params.items():
                data_to_save[f"{key}_{param_name}"] = param_value.item()

        # Calculate elapsed time and estimate remaining time
        elapsed_time = time.time() - start_time
        iterations_done = iteration + 1
        iterations_left = num_iterations - iterations_done
        time_per_iteration = elapsed_time / iterations_done
        estimated_time_remaining = iterations_left * time_per_iteration

        # Format the elapsed time and estimated remaining time
        formatted_elapsed_time = red_text(format_time(elapsed_time))
        formatted_remaining_time = red_text(format_time(estimated_time_remaining))

        # Print the parameters, losses, and timing information in a single line
        param_strings = [f"{key}_{param_name}: {param_value.item()}" for key, params in
                         optimized_hyperparameters.items() for param_name, param_value in params.items()]
        lambda_strings = [f"Lambda_hyp_{fm}: {lambda_val.item()}" for fm, lambda_val in optimized_lambda_hyp.items()]
        fm_loss_strings = [f"fm{fm}_Loss: {fm_losses[fm]}" for fm in failure_modes]
        print(f"Iteration {iteration + 1}, Total_Loss: {total_loss}, " +
              ", ".join(fm_loss_strings) +
              f", Elapsed Time: {formatted_elapsed_time}, Estimated Time Remaining: {formatted_remaining_time}, " +
              ", ".join(param_strings + lambda_strings))

        # Save the data to a TSV file
        if not header_written:
            header = data_to_save.keys()
            np.savetxt(os.path.join(main_folder_path, f"{file_name}.tsv"), [list(header)], delimiter="\t", fmt="%s")
            header_written = True

        with open(os.path.join(main_folder_path, f"{file_name}.tsv"), "a") as f:
            np.savetxt(f, [list(data_to_save.values())], delimiter="\t", fmt="%s")

        # Save intermediate models and data at the save interval
        if (iteration + 1) % save_interval == 0:
            iteration_folder = os.path.join(main_folder_path, f"iteration_{iteration + 1}")
            os.makedirs(iteration_folder, exist_ok=True)

            tsv_path = os.path.join(iteration_folder, f"{file_name}_{iteration + 1}.tsv")
            model_path = os.path.join(iteration_folder, f"{file_name}_{iteration + 1}.pth")

            with open(tsv_path, "w") as f:
                header = data_to_save.keys()
                np.savetxt(f, [list(header)], delimiter="\t", fmt="%s")
                np.savetxt(f, [list(data_to_save.values())], delimiter="\t", fmt="%s")

            torch.save({
                'optimized_hyperparameters': optimized_hyperparameters,
                'optimized_lambda_hyp': optimized_lambda_hyp
            }, model_path)

    return header_written


########################################################################################################################
def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def save_and_print_parameters(iteration, num_iterations, losses, flat_parameters_initial, metadata, main_folder_path,
                              file_name, save_interval=1000, print_interval=1, header_written=False,
                              lagrange_multiplier=None, start_time=None):
    # Only execute the following block at the specified print interval
    if (iteration + 1) % print_interval == 0:
        total_loss, neg_elbo, cox_loss, gp1_loss, gp2_loss, kl_loss, pi_constraint = losses[-1]

        # Reconstruct hyperparameters
        optimized_params = reconstruct_hyperparameters_elbo(flat_parameters_initial.detach(), metadata)
        optimized_pi_hat = optimized_params['pi_hat']
        optimized_b = optimized_params['b']
        optimized_psi = optimized_params['psi']
        optimized_alpha_hat = optimized_params['alpha_hat']
        optimized_beta = optimized_params['beta']
        optimized_gamma = optimized_params['gamma']
        optimized_lagrange_multiplier = lagrange_multiplier.detach()

        # Prepare data for saving
        data_to_save = {
            'Iteration': iteration + 1,
            'Total loss': total_loss,
            'Negative Elbo': neg_elbo,
            'Cox loss': cox_loss,
            'GP1 loss': gp1_loss,
            'GP2 loss': gp2_loss,
            'KL loss': kl_loss,
            'Pi_constraint loss': pi_constraint,
        }

        # Dynamically add parameters to the data to save
        for key, params in optimized_params.items():
            for param_name, param_tensor in params.items():
                if param_tensor.numel() == 1:
                    data_to_save[f"{key}_{param_name}"] = param_tensor.item()
                else:
                    data_to_save[f"{key}_{param_name}"] = param_tensor.tolist()

        # Calculate elapsed time and estimate remaining time
        elapsed_time = time.time() - start_time
        iterations_done = iteration + 1
        iterations_left = num_iterations - iterations_done
        time_per_iteration = elapsed_time / iterations_done
        estimated_time_remaining = iterations_left * time_per_iteration

        # Format the elapsed time and estimated remaining time
        formatted_elapsed_time = red_text(format_time(elapsed_time))
        formatted_remaining_time = red_text(format_time(estimated_time_remaining))

        # Prepare parameter strings for printing
        param_strings = [
            f"{key}_{param_name}: {param_tensor.item() if param_tensor.numel() == 1 else param_tensor.tolist()}"
            for key, params in optimized_params.items() for param_name, param_tensor in params.items()
        ]

        # Print the iteration, elapsed time, remaining time, and parameters
        print(
            f"Iteration {iteration + 1}, Elapsed Time: {formatted_elapsed_time}, "
            f"Estimated Time Remaining: {formatted_remaining_time}, Total Loss: {total_loss}, "
            f"Negative Elbo: {neg_elbo}, Cox Loss: {cox_loss}, GP1 Loss: {gp1_loss}, "
            f"GP2 Loss: {gp2_loss}, KL Loss: {kl_loss}, Lagrange Multiplier: {optimized_lagrange_multiplier.item()}, "
            f"Pi_constraint loss: {pi_constraint}, " + ", ".join(param_strings)
        )

        # Save the data to a TSV file
        if not header_written:
            header = data_to_save.keys()
            np.savetxt(os.path.join(main_folder_path, f"{file_name}.tsv"), [list(header)], delimiter="\t", fmt="%s")
            header_written = True

        with open(os.path.join(main_folder_path, f"{file_name}.tsv"), "a") as f:
            np.savetxt(f, [list(data_to_save.values())], delimiter="\t", fmt="%s")

        # Save intermediate models and data at the save interval
        if (iteration + 1) % save_interval == 0:
            iteration_folder = os.path.join(main_folder_path, f"iteration_{iteration + 1}")
            os.makedirs(iteration_folder, exist_ok=True)

            tsv_path = os.path.join(iteration_folder, f"{file_name}_{iteration + 1}.tsv")
            model_path = os.path.join(iteration_folder, f"{file_name}_{iteration + 1}.pth")

            with open(tsv_path, "w") as f:
                header = data_to_save.keys()
                np.savetxt(f, [list(header)], delimiter="\t", fmt="%s")
                np.savetxt(f, [list(data_to_save.values())], delimiter="\t", fmt="%s")

            torch.save({
                'optimized_beta': optimized_beta,
                'optimized_pi_hat': optimized_pi_hat,
                'optimized_b': optimized_b,
                'optimized_psi': optimized_psi,
                'optimized_alpha_hat': optimized_alpha_hat,
                'optimized_gamma': optimized_gamma,
            }, model_path)

    return header_written


def log_message(message):
    print(message)  # Keeps the print output in the console
    logging.info(message)


def save_and_print_gp_hyperparams_test(iteration, num_iterations, losses, flat_parameters_initial,
                                       loaded_hyperparameters, test_keys, optimized_lambda_hyp,
                                       main_folder_path, file_name, save_interval=10000,
                                       print_interval=1, header_written=False,
                                       failure_modes=None, start_time=None):
    # Only execute the following block at the specified print interval
    if (iteration + 1) % print_interval == 0:
        # Reconstruct hyperparameters and lambda_hyp for each failure mode
        optimized_hyperparameters = reconstruct_hyperparameters_test(
            flat_parameters_initial.detach(), loaded_hyperparameters, test_keys)

        # Prepare data for saving
        total_loss = losses[-1][0]
        data_to_save = {
            'Iteration': iteration + 1,
            'Total_Loss': total_loss,
        }

        # Dynamically add losses for each failure mode
        fm_losses = {}
        for i, fm in enumerate(failure_modes):
            fm_losses[fm] = losses[-1][i + 1]
            data_to_save[f"fm{fm}_Loss"] = fm_losses[fm]

        # Include the hyperparameters in the data to save
        for key, params in optimized_hyperparameters.items():
            for param_name, param_value in params.items():
                data_to_save[f"{key}_{param_name}"] = param_value.item()

        # Calculate elapsed time and estimate remaining time
        elapsed_time = time.time() - start_time
        iterations_done = iteration + 1
        iterations_left = num_iterations - iterations_done
        time_per_iteration = elapsed_time / iterations_done
        estimated_time_remaining = iterations_left * time_per_iteration

        # Format the elapsed time and estimated remaining time
        formatted_elapsed_time = red_text(format_time(elapsed_time))
        formatted_remaining_time = red_text(format_time(estimated_time_remaining))

        # Print the parameters, losses, and timing information in a single line
        param_strings = [f"{key}_{param_name}: {param_value.item()}" for key, params in
                         optimized_hyperparameters.items() for param_name, param_value in params.items()]

        fm_loss_strings = [f"fm{fm}_Loss: {fm_losses[fm]}" for fm in failure_modes]
        print(f"Iteration {iteration + 1}, Total_Loss: {total_loss}, " +
              ", ".join(fm_loss_strings) +
              f", Elapsed Time: {formatted_elapsed_time}, Estimated Time Remaining: {formatted_remaining_time}, " +
              ", ".join(param_strings))

        # Save the data to a TSV file
        if not header_written:
            header = data_to_save.keys()
            np.savetxt(os.path.join(main_folder_path, f"{file_name}.tsv"), [list(header)], delimiter="\t", fmt="%s")
            header_written = True

        with open(os.path.join(main_folder_path, f"{file_name}.tsv"), "a") as f:
            np.savetxt(f, [list(data_to_save.values())], delimiter="\t", fmt="%s")

        # Save intermediate models and data at the save interval
        if (iteration + 1) % save_interval == 0:
            iteration_folder = os.path.join(main_folder_path, f"iteration_{iteration + 1}")
            os.makedirs(iteration_folder, exist_ok=True)

            tsv_path = os.path.join(iteration_folder, f"{file_name}_{iteration + 1}.tsv")
            model_path = os.path.join(iteration_folder, f"{file_name}_{iteration + 1}.pth")

            with open(tsv_path, "w") as f:
                header = data_to_save.keys()
                np.savetxt(f, [list(header)], delimiter="\t", fmt="%s")
                np.savetxt(f, [list(data_to_save.values())], delimiter="\t", fmt="%s")

            torch.save({
                'optimized_hyperparameters': optimized_hyperparameters,
                'optimized_lambda_hyp': optimized_lambda_hyp
            }, model_path)

    return header_written


def save_and_print_gp_hyperparams_test_2(iteration, num_iterations, losses, flat_parameters_initial,
                                         loaded_hyperparameters, test_keys, optimized_lambda_hyp,
                                         main_folder_path, file_name, hyper_test, save_interval=10000,
                                         print_interval=1, header_written=False,
                                         failure_modes=None, start_time=None):
    # Only execute the following block at the specified print interval
    if (iteration + 1) % print_interval == 0:
        # Reconstruct hyperparameters and lambda_hyp for each failure mode
        optimized_hyperparameters = reconstruct_hyperparameters_test_2(
            flat_parameters_initial.detach(), loaded_hyperparameters, test_keys, hyper_test)

        # Prepare data for saving
        total_loss = losses[-1][0]
        data_to_save = {
            'Iteration': iteration + 1,
            'Total_Loss': total_loss,
        }

        # Dynamically add losses for each failure mode
        fm_losses = {}
        for i, fm in enumerate(failure_modes):
            fm_losses[fm] = losses[-1][i + 1]
            data_to_save[f"fm{fm}_Loss"] = fm_losses[fm]

        # Include the hyperparameters in the data to save
        for key, params in optimized_hyperparameters.items():
            for param_name, param_value in params.items():
                data_to_save[f"{key}_{param_name}"] = param_value.item()

        # Calculate elapsed time and estimate remaining time
        elapsed_time = time.time() - start_time
        iterations_done = iteration + 1
        iterations_left = num_iterations - iterations_done
        time_per_iteration = elapsed_time / iterations_done
        estimated_time_remaining = iterations_left * time_per_iteration

        # Format the elapsed time and estimated remaining time
        formatted_elapsed_time = red_text(format_time(elapsed_time))
        formatted_remaining_time = red_text(format_time(estimated_time_remaining))

        # Print the parameters, losses, and timing information in a single line
        param_strings = [f"{key}_{param_name}: {param_value.item()}" for key, params in
                         optimized_hyperparameters.items() for param_name, param_value in params.items()]

        fm_loss_strings = [f"fm{fm}_Loss: {fm_losses[fm]}" for fm in failure_modes]
        print(f"Iteration {iteration + 1}, Total_Loss: {total_loss}, " +
              ", ".join(fm_loss_strings) +
              f", Elapsed Time: {formatted_elapsed_time}, Estimated Time Remaining: {formatted_remaining_time}, " +
              ", ".join(param_strings))

        # Save the data to a TSV file
        if not header_written:
            header = data_to_save.keys()
            np.savetxt(os.path.join(main_folder_path, f"{file_name}.tsv"), [list(header)], delimiter="\t", fmt="%s")
            header_written = True

        with open(os.path.join(main_folder_path, f"{file_name}.tsv"), "a") as f:
            np.savetxt(f, [list(data_to_save.values())], delimiter="\t", fmt="%s")

        # Save intermediate models and data at the save interval
        if (iteration + 1) % save_interval == 0:
            iteration_folder = os.path.join(main_folder_path, f"iteration_{iteration + 1}")
            os.makedirs(iteration_folder, exist_ok=True)

            tsv_path = os.path.join(iteration_folder, f"{file_name}_{iteration + 1}.tsv")
            model_path = os.path.join(iteration_folder, f"{file_name}_{iteration + 1}.pth")

            with open(tsv_path, "w") as f:
                header = data_to_save.keys()
                np.savetxt(f, [list(header)], delimiter="\t", fmt="%s")
                np.savetxt(f, [list(data_to_save.values())], delimiter="\t", fmt="%s")

            torch.save({
                'optimized_hyperparameters': optimized_hyperparameters,
                'optimized_lambda_hyp': optimized_lambda_hyp
            }, model_path)

    return header_written


def save_and_print_parameters_2(iteration, num_iterations, losses, flat_parameters_initial, metadata, main_folder_path,
                                file_name, save_interval=1000, print_interval=1, header_written=False,
                                lagrange_multiplier=None, start_time=None):
    # Only execute the following block at the specified print interval
    if (iteration + 1) % print_interval == 0:
        total_loss, neg_elbo, cox_loss, gp1_loss, gp2_loss, kl_loss, pi_constraint, kl_b_rho_loss = losses[-1]

        # Reconstruct hyperparameters
        optimized_params = reconstruct_hyperparameters_elbo_2(flat_parameters_initial.detach(), metadata)
        optimized_pi_hat = optimized_params['pi_hat']
        optimized_mu_b_hat = optimized_params['mu_b_hat']
        optimized_sigma_b_hat = optimized_params['sigma_b_hat']
        optimized_alpha_rho_hat = optimized_params['alpha_rho_hat']
        optimized_beta_rho_hat = optimized_params['beta_rho_hat']
        optimized_alpha_hat = optimized_params['alpha_hat']
        optimized_beta = optimized_params['beta']
        optimized_gamma = optimized_params['gamma']
        optimized_lagrange_multiplier = lagrange_multiplier.detach()

        # Prepare data for saving
        data_to_save = {
            'Iteration': iteration + 1,
            'Total loss': total_loss,
            'Negative Elbo': neg_elbo,
            'Cox loss': cox_loss,
            'GP1 loss': gp1_loss,
            'GP2 loss': gp2_loss,
            'KL loss': kl_loss,
            'Pi_constraint loss': pi_constraint,
            'kl_b_rho loss': kl_b_rho_loss
        }

        # Dynamically add parameters to the data to save
        for key, params in optimized_params.items():
            for param_name, param_tensor in params.items():
                if param_tensor.numel() == 1:
                    data_to_save[f"{key}_{param_name}"] = param_tensor.item()
                else:
                    data_to_save[f"{key}_{param_name}"] = param_tensor.tolist()

        # Calculate elapsed time and estimate remaining time
        elapsed_time = time.time() - start_time
        iterations_done = iteration + 1
        iterations_left = num_iterations - iterations_done
        time_per_iteration = elapsed_time / iterations_done
        estimated_time_remaining = iterations_left * time_per_iteration

        # Format the elapsed time and estimated remaining time
        formatted_elapsed_time = red_text(format_time(elapsed_time))
        formatted_remaining_time = red_text(format_time(estimated_time_remaining))

        # Prepare parameter strings for printing
        param_strings = [
            f"{key}_{param_name}: {param_tensor.item() if param_tensor.numel() == 1 else param_tensor.tolist()}"
            for key, params in optimized_params.items() for param_name, param_tensor in params.items()
        ]

        # Print the iteration, elapsed time, remaining time, and parameters
        print(
            f"Iteration {iteration + 1}, Elapsed Time: {formatted_elapsed_time}, "
            f"Estimated Time Remaining: {formatted_remaining_time}, Total Loss: {total_loss}, "
            f"Negative Elbo: {neg_elbo}, Cox Loss: {cox_loss}, GP1 Loss: {gp1_loss}, "
            f"GP2 Loss: {gp2_loss}, KL Loss: {kl_loss}, Lagrange Multiplier: {optimized_lagrange_multiplier.item()}, "
            f"Pi_constraint loss: {pi_constraint},'kl_b_rho loss': {kl_b_rho_loss} , " + ", ".join(param_strings)
        )

        # Save the data to a TSV file
        if not header_written:
            header = data_to_save.keys()
            np.savetxt(os.path.join(main_folder_path, f"{file_name}.tsv"), [list(header)], delimiter="\t", fmt="%s")
            header_written = True

        with open(os.path.join(main_folder_path, f"{file_name}.tsv"), "a") as f:
            np.savetxt(f, [list(data_to_save.values())], delimiter="\t", fmt="%s")

        # Save intermediate models and data at the save interval
        if (iteration + 1) % save_interval == 0:
            iteration_folder = os.path.join(main_folder_path, f"iteration_{iteration + 1}")
            os.makedirs(iteration_folder, exist_ok=True)

            tsv_path = os.path.join(iteration_folder, f"{file_name}_{iteration + 1}.tsv")
            model_path = os.path.join(iteration_folder, f"{file_name}_{iteration + 1}.pth")

            with open(tsv_path, "w") as f:
                header = data_to_save.keys()
                np.savetxt(f, [list(header)], delimiter="\t", fmt="%s")
                np.savetxt(f, [list(data_to_save.values())], delimiter="\t", fmt="%s")

            torch.save({
                'optimized_beta': optimized_beta,
                'optimized_pi_hat': optimized_pi_hat,
                'optimized_mu_b_hat': optimized_mu_b_hat,
                'optimized_sigma_b_hat': optimized_sigma_b_hat,
                'optimized_alpha_rho_hat': optimized_alpha_rho_hat,
                'optimized_beta_rho_hat': optimized_beta_rho_hat,
                'optimized_alpha_hat': optimized_alpha_hat,
                'optimized_gamma': optimized_gamma,
            }, model_path)

    return header_written
