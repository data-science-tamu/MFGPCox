import numpy as np
import scipy.integrate as spi
import pandas as pd
from lifelines.utils import to_long_format
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import torch




beta = [1.05,0.25]
v = 1.05
lamb = 0.0001
sigma2 = 0.01
gam = 0.2


# this function get time and coefficient matrix and return sensor signal (without noise), Basis @ Coef


# def zt(t, a_b_c_matrix):
#     if isinstance(t, np.ndarray):
#         t_matrix = np.c_[np.ones([len(t), 1]), t, t*t]
#     else:
#         t_matrix = [1, t, t*t]
#     zt = np.matmul(t_matrix, a_b_c_matrix)
#     return zt





def zt_1(t, a_b_c_matrix):
    # Ensure t is treated as an array for consistent handling
    t = np.atleast_1d(t)
    # Construct t_matrix
    t_matrix = np.c_[np.ones_like(t), 0.3 * (t ** 0.6 * np.sin(t)),  t ** 2]
    # Matrix multiplication
    zt = np.dot(t_matrix, a_b_c_matrix)
    return zt.squeeze()  # Ensure scalar output for scalar input




def zt_2(t, a_b_c_matrix):
    # Ensure t is treated as an array for consistent handling
    t = np.atleast_1d(t)
    # Construct t_matrix
    t_matrix = np.c_[np.ones_like(t), t ** 1.5, t ** 2]
    # Matrix multiplication
    zt = np.dot(t_matrix, a_b_c_matrix)
    return zt.squeeze()  # Ensure scalar output for scalar input


# Returns a scalar value y, which is the degradation signal at time t- eq(23) in the paper without noise term
def degradation_func(t, b):
    T_matrix = np.array([[1, t, t**2]]).T
    y = np.matmul(b, T_matrix)[0]
    return y


# eq(25) within exp (no sqrt in the paper???)
def non_linear_fun(t,x,b1,b2):
    result=np.sqrt(beta[0]*degradation_func(t,b1)**2+beta[1]*degradation_func(t,b2)**2)
    return result


# hazard eq(25)
def ht_func(t,x,b1,b2):

    exp1=np.exp(non_linear_fun(t,x,b1,b2))
    ht=lamb*v*(t**(v-1))*exp1
    return ht




# survival probavility
def St(t, x, b1, b2):
    if ht_func(t, x, b1, b2) is None:
        return None
    else:
        St=np.exp(np.negative(spi.quad(ht_func, 0, t, args=(x, b1, b2))[0]))
        return St


def ft(t, x, b1, b2):
    if St(t, x, b1, b2) is None:
        return None
    else:
        ft_func = ht_func(t, x, b1, b2)*St(t, x, b1, b2)
        return ft_func


def St_cond(t, tstar, x, b1, b2):
    St_cond = St(t, x, b1, b2)/St(tstar, x, b1, b2)
    return St_cond



def load_data(Normalize=True):

    data = pd.read_csv('Training/training_no_noise.csv')
    variable = pd.read_csv('Training/status_14-01-2021.csv')

    status = variable['status']
    base_df = to_long_format(data, duration_col="obstime")

    base_df[['start']] = base_df[['stop']] - 1
    base_df['died'] = np.zeros([len(base_df), 1], dtype=int)
    Unit = base_df['id'].unique()
    ts = 0
    min = 0
    max = 0
    for item in Unit:
        temp = base_df.loc[base_df['id'] == item]
        ir = len(temp)
        base_df.loc[(ts):(ts + ir - 2), 'died'] = 0
        base_df.loc[ts + ir - 1, 'died'] = int(status[item - 1])
        ts = ts + ir

    base_df.to_csv('base_data_simu.csv')
    if Normalize:
        cols_normalize = base_df.columns.difference(['id', 'start', 'stop', 'died'])
        min = base_df[cols_normalize].min()
        max = base_df[cols_normalize].max()
        norm_train_data = pd.DataFrame((base_df[cols_normalize] - min)/(max - min),
                                       columns=cols_normalize, index=base_df.index)
        join_data = base_df[base_df.columns.difference(cols_normalize)].join(norm_train_data)
        train_data = join_data.reindex(columns=base_df.columns)
    else:
        train_data = base_df

    return train_data, min, max

def Compare_hazard_plot(data, id, H0t, model):
    id_data = data[data['id'] == id]
    id_data = id_data[id_data.columns.difference(['id', 'start', 'stop', 'died', 'x'])]
    risk_pre = model(torch.tensor(id_data.values).float())
    risk_pre = np.exp(risk_pre.detach().numpy())
    hazard_pre = []
    hazard_true = []

    h0t_true = []
    h0t_pre = []

    a_b_matrix_true = pd.read_csv('./Training/True_Beta_new.txt', header=None, sep=' ')
    a_b_matrix_true = a_b_matrix_true.values

    for t in range(0, len(id_data)):
        risk=non_linear_fun(t+1,0,a_b_matrix_true[id - 1, 0:3],a_b_matrix_true[id - 1,3:6])
        hazard_pre.append(risk_pre[t] * H0t(t+1))
        hazard_true.append(np.exp(risk) * lamb * v * ((t+1)**(0.05)))


    t = np.linspace(1, len(id_data), len(id_data))
    plt.plot(t, hazard_pre, label='Estimated')
    plt.plot(t, hazard_true, label='True')
    plt.legend()
    plt.title('Hazard Function')
    plt.show()


def hazard(data, id, H0t, model):
    id_data = data[data['id'] == id]
    id_data = id_data[id_data.columns.difference(['id', 'start', 'stop', 'died', 'x'])]
    risk_pre = model(torch.tensor(id_data.values).float())
    risk_pre = np.exp(risk_pre.detach().numpy())
    hazard_pre = []

    for t in range(0, len(id_data)):
        hazard_pre.append(risk_pre[t] * H0t(t + 1))


    t = np.linspace(1, len(id_data), len(id_data))
    plt.plot(t, hazard_pre, label='Estimated')
    plt.legend()
    plt.title('Hazard Function')
    plt.show()


def compare_pre_result(tstar_time,unit, model,node_in, smoothed_H0t, min, max):
    upper = 191
    degree = 10

    x, w = np.polynomial.legendre.leggauss(degree)
    # a_b_matrix_true = pd.read_csv('./Test_simulation/True_Beta_new.txt', sep=' ',header=None)
    # a_b_matrix_true = a_b_matrix_true.values

    sigma20 = pd.read_csv('Training/sigma20.csv', index_col=0)
    sigma20_matrix = sigma20.values
    MUB0 = pd.read_csv('Training/MUB0.csv', index_col=0)
    MUB0_matrix = MUB0.values.reshape(node_in, 3)
    SIGMAB0 = pd.read_csv('Training/SIGMAB0.csv', index_col=0)
    SIGMAB0_matrix = SIGMAB0.values.reshape(node_in, 9)

    # input test data
    column_test = [2, 3]
    test_data = pd.read_csv('Test_simulation/processed_test_data_all.csv')
    IDs=test_data['ID'].unique()
    test_data = test_data.values
    test_index = test_data[:, 0:2]
    test_data_new = np.hstack((test_index, (test_data[:, column_test])))

    samples = 1000
    est_mrl = np.zeros([100, 6])
    true_mrl = np.zeros([100,6])
    est_sz=[]
    est_sz_lower = []
    est_sz_upper = []

    tstar=[tstar_time]
    for test_unit in [unit]:
        print(test_unit)
        for g in range(0, len(tstar)):
            example = test_data_new[test_data_new[:, 0] == test_unit, :].astype(float)

            # Beyesian update (parameters a and b)
            MU = []
            SIGMA = []
            time = example[:, 1]
            time = time[0:tstar[g]]
            time2 = time * time
            X = np.concatenate((np.ones([len(time), 1]), time.reshape(len(time), 1)), 1)
            X = np.hstack((X, time2.reshape(len(time), 1)))

            a_matrix = []
            b_matrix = []
            c_matrix = []

            # eq 17 and 18 from the paper
            for q in range(0, len(column_test)):
                Y = example[0:tstar[g], q + 2]
                SIGMAB0_M = SIGMAB0_matrix[q].reshape(3, 3)
                SIGMA.append(
                    np.linalg.inv(np.matmul(np.transpose(X), X) / sigma20_matrix[q][0] + np.linalg.inv(SIGMAB0_M)))

                X_Y = np.matmul(np.transpose(X), Y).reshape(3, 1)
                N_matrix = np.matmul(np.linalg.inv(SIGMAB0_M), MUB0_matrix[q].reshape(3, 1)) + X_Y / sigma20_matrix[q][
                    0]
                MU.append(np.matmul(SIGMA[q], N_matrix))

                a_est = np.empty(shape=[samples, 1])
                b_est = np.empty(shape=[samples, 1])
                c_est = np.empty(shape=[samples, 1])
                for e in range(0, samples):
                    a_est[e], b_est[e], c_est[e] = np.random.multivariate_normal(MU[q].T[0], SIGMA[q])
                a_matrix.append(a_est)
                b_matrix.append(b_est)
                c_matrix.append(c_est)
            a_matrix = np.array(a_matrix).T
            b_matrix = np.array(b_matrix).T
            c_matrix = np.array(c_matrix).T

            #         Calculate conditional survival function

            est_sz_matrix=np.zeros([samples,len(range(tstar[g], upper))])

            for l in range(0,samples):
                a_b_c_matrix=np.vstack((np.vstack((a_matrix[0][l,:],b_matrix[0][l,:])),c_matrix[0][l,:]))
#                Ht=[]
                k=0
                for m in range(tstar[g],upper):

                    per_t=np.linspace(tstar[g],m,1000)

                    # test_input=zt(per_t,a_b_c_matrix)
                    col_1 = zt_1(per_t, a_b_c_matrix[:, 0])  # use the parameters for q=0
                    col_2 = zt_2(per_t, a_b_c_matrix[:, 1])  # use the parameters for q=1

                    # Stack columns horizontally -> shape (1000, 2)
                    test_input = np.column_stack((col_1, col_2))
                    test_input = pd.DataFrame(test_input, columns=['degradation1', 'degradation2'])
                    out_temp = model(torch.tensor(test_input.values).float())
                    out_temp = out_temp.detach().numpy()

                    h0t = smoothed_H0t(per_t)
                    h0t[h0t < 0] = 0

                    out_integral = sum(((h0t.reshape([1000, 1])) * np.exp(out_temp)) * ((m - tstar[g]) / 999))
                    est_sz_matrix[l,k]=np.exp(np.negative(out_integral))
                    k=k+1

            # this is survivals from t* to 100 (upper)
            est_sz.append(est_sz_matrix.sum(axis=0)/samples)

            est_sz_lower.append(np.quantile(est_sz_matrix, 0.025, axis=0))
            est_sz_upper.append(np.quantile(est_sz_matrix, 0.975, axis=0))

            mrl = np.zeros([samples, 1])

            for l in range(0, samples):
                a_b_c_matrix = np.vstack((np.vstack((a_matrix[0][l, :], b_matrix[0][l, :])), c_matrix[0][l, :]))
                for m in range(0, degree):
                    t_new = x[m] * (upper - tstar[g]) / 2 + (upper + tstar[g]) / 2
                    per_t = np.linspace(tstar[g], t_new, 1000)


                    col_1 = zt_1(per_t, a_b_c_matrix[:, 0])  # use the parameters for q=0
                    col_2 = zt_2(per_t, a_b_c_matrix[:, 1])  # use the parameters for q=1

                    # Stack columns horizontally -> shape (1000, 2)
                    test_input = np.column_stack((col_1, col_2))
                    test_input = pd.DataFrame(test_input, columns=['degradation1', 'degradation2'])
                    # test_input = zt(per_t, a_b_c_matrix)
                    # test_input = pd.DataFrame(test_input, columns=['degradation1', 'degradation2'])
                    # # test_input = (test_input - test_input.min()) / (test_input.max() - test_input.min())

                    h0t = smoothed_H0t(per_t)
                    h0t[h0t < 0] = 0
                    out_temp = model(torch.tensor(test_input.values).float())
                    out_temp = out_temp.detach().numpy()

                    out_integral = sum(((h0t.reshape([1000, 1])) * np.exp(out_temp)) * ((t_new - tstar[g]) / 999))

                    mrl[l] = mrl[l] + np.exp((np.negative(out_integral))) * w[m] * (upper - tstar[g]) / 2

            # est_mrl[test_unit-1, g] = sum(mrl / samples)
            mrl_est = sum(mrl / samples)
    #
    # ###### true mrl #####
    #
    #
    #
    #         for i in range(0, degree):
    #             true_mrl[test_unit-1, g] = true_mrl[test_unit-1, g] + St_cond(
    #                 x[i] * (upper - tstar[g]) / 2 + (upper + tstar[g]) / 2, tstar[g], 1, a_b_matrix_true[test_unit - 1, 0:3],
    #                 a_b_matrix_true[test_unit - 1, 3:6]) * w[i] * (upper - tstar[g]) / 2

            #### true conditional survival function
    # St_cond_values = []
    # for test_unit in [74]:
    #     for g in range(0, len(tstar)):
    #         St_cond_value = []
    #         for m in range(tstar[g], upper):
    #             #            St_cond_value.append(St_cond(m,tstar[g],a_b_matrix_true[test_unit-1][0],a_b_matrix_true[test_unit-1][1],a_b_matrix_true[test_unit-1][2],a_b_matrix_true[test_unit-1][3]))
    #             St_cond_value.append(
    #                 St_cond(m, tstar[g], 1, a_b_matrix_true[test_unit - 1, 0:3], a_b_matrix_true[test_unit - 1, 3:6]))
    #
    #         St_cond_values.append(St_cond_value)



    # return est_sz_matrix,St_cond_values

    return est_sz,est_sz_lower, est_sz_upper, mrl_est, est_sz_matrix



