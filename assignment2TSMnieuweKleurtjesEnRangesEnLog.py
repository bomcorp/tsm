import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize, stats
import numdifftools as nd
import datetime
import csv
import time


def initializize_optimization():
    # Initialize parameters of h_{t+1}
    omega_ini = 1
    phi_ini = 0.9
    sig2_eta_ini = 1

    theta_ini = [omega_ini, phi_ini, sig2_eta_ini]

    # Set bounds for parameters
    omega_bnds = (-np.inf, np.inf)
    phi_bnds = (0, 0.9999)
    # if ub is set to 1, phi might become 1 and initialization of a in KF will be -inf
    sig2_eta_bnds = (0, np.inf)

    theta_bnds = [omega_bnds, phi_bnds, sig2_eta_bnds]

    options = {'eps': 1e-09,
               'disp': True,
               'maxiter': 200}

    return theta_ini, theta_bnds, options


def Kalman_filter(y, H, Z, R, mean_u, T, Q, omega, beta=0, SV=0):
    I = len(y)
    a = np.zeros(I)
    att = np.zeros(I)
    a[0] = omega / (1 - T[0])  # unconditional expectation AR(1) model
    if SV == 0:
        SV = np.zeros(I)
     
    P = np.zeros(I)
    P[0] = Q[0] / (1 - T[0] ** 2)  # unconditional variance AR(1) model
    Ptt = np.zeros(I)
    v = np.zeros(I)
    F = np.zeros(I)
    K = np.zeros(I)
    for i in range(I):
        v[i] = y[i] - Z[i] * a[i] - mean_u - beta * SV[i]  # hij valt hierover
        F[i] = Z[i] * P[i] * np.transpose(Z[i]) + H[i]
        K[i] = T[i] * P[i] * np.transpose(Z[i]) * F[i] ** (-1)
        att[i] = a[i] + P[i] * np.transpose(Z[i]) * F[i] ** (-1) * v[i]
        Ptt[i] = P[i] - P[i] * np.transpose(Z[i]) * F[i] ** (-1) * Z[i] * P[i]
        if i < I - 1:
            a[i + 1] = T[i] * att[i] + omega
            P[i + 1] = T[i] * Ptt[i] * np.transpose(T[i]) + R[i] * Q[i] * np.transpose(R[i])
    return F, K, v, a + mean_u, P, T

def calc_r(v, F, L, I):
    r = np.zeros(I)

    for i in range(I - 1, 0, -1):
        r[i - 1] = v[i] / F[i] + L[i] * r[i]

    return r


def calc_alpha_hat(r, a, P, v, F, L, I):
    alpha_hat = np.zeros(I)
    alpha_hat[0] = a[0] + P[0] * (v[0]/F[0] + L[0]*r[0])
    for i in range(1, I):
        alpha_hat[i] = a[i] + P[i] * r[i - 1]

    return alpha_hat


def calc_N(F, L, I):
    N = np.zeros(I)

    for i in range(I - 1, 0, -1):
        N[i - 1] = F[i] ** (-1) + L[i] ** 2 * N[i]

    return N


def calc_V(N, P, F, L, I):
    V = np.zeros(I)
    V[0] = P[0] - P[0]**2 * (F[0] ** (-1) + L[0] ** 2 * N[0])
    for i in range(I):
        V[i] = P[i] - P[i] ** 2 * N[i - 1]

    return V


def Kalman_state_smoothing(y, a, v, P, F, K, T, Z):
    I = len(y)
    L = T - K * Z

    r = calc_r(v, F, L, I)
    alpha_hat = calc_alpha_hat(r, a, P, v, F, L, I)
    N = calc_N(F, L, I)
    V = calc_V(N, P, F, L, I)

    return L, V, N, alpha_hat, r


def plot_size(y, x):
    figure_dim = plt.rcParams["figure.figsize"]
    figure_dim[0] = y
    figure_dim[1] = x
    plt.rcParams["figure.figsize"] = figure_dim

    return figure_dim


def fig_14_5_i(y):
    I = len(y)
    plt.plot(y, color='grey')
    plt.plot(np.zeros(I), color='k')
    plt.title('daily, mean-corrected, log-returns of exchangerates')
    figure_dim = plot_size(10, 4)
    plt.show()


def fig_14_5_ii_p1(x):
    plt.plot(x, linestyle="", marker="o", color='k', markersize=2)
    plt.title('log-daily, mean-corrected, squared-log-returns of exchangerates')
    figure_dim = plot_size(10, 4)
    plt.show()


def fig_14_5_ii_p2(x, alpha_hat_adj, a):
    plt.plot(x, linestyle="", marker="o", color='k', markersize=2)
    plt.plot(alpha_hat_adj, color='red', label='alpha_hat smoothed state')
    plt.plot(a, label='a, filtered state')
    plt.legend()
    plt.title('smoothed estimate')
    figure_dim = plot_size(10, 4)
    plt.show()


def llik_fun_SV(theta_ini, y, I, RV):
    omega = theta_ini[0]
    phi = theta_ini[1]
    sig2_eta = theta_ini[2]
    beta = 0
    if len(theta_ini) == 4:
        beta = theta_ini[3]

    sig2_u = (np.pi ** 2) / 2  # u = log(eps_t**2)
    mean_u = -1.27

    Q = np.ones(I) * sig2_eta
    T = np.ones(I) * phi
    Z = np.ones(I)
    H = np.ones(I) * sig2_u
    R = np.ones(I)

    F, K, v, a, P, T = Kalman_filter(y, H, Z, R, mean_u, T, Q, omega, beta, RV)

    l = -(I / 2) * np.log(2 * np.pi) - (1 / 2) * np.sum(np.log(F) + v ** 2 / F)

    # Check for negative F
    for i in range(len(F) - 1):
        if F[i] <= 0:
            print('Negative prediction error variance')

    return -np.mean(l)


def loadDataOxford(fileName, amtHeaderRows=1, columnOfData=5, delimiter=','):
    data = []
    counter = 0
    with open(fileName, 'rt') as csvfile:
        dataset = csv.reader(csvfile, dialect='excel', delimiter=delimiter, quotechar='|')
        for row in dataset:
            if counter < amtHeaderRows:  # skip header row
                counter += 1
            else:
                if row[columnOfData - 1] == 'NA' or row[columnOfData - 1] == 'null':
                    pass
                else:
                    if row[1] == ".SPX":
                        data.append(float(row[columnOfData-1]))
    return data

def loadDatasv(fileName, amtHeaderRows=1, columnOfData=0, delimiter=','):
    data = []
    counter = 0
    with open(fileName, 'rt') as csvfile:
        dataset = csv.reader(csvfile, dialect='excel', delimiter=delimiter, quotechar='|')
        for row in dataset:
            if counter < amtHeaderRows:  # skip header row
                counter += 1
            else:
                if row[columnOfData - 1] == 'NA' or row[columnOfData - 1] == 'null':
                    pass
                else:
                    data.append(float(row[columnOfData-1]))
    return data

def convertToLogDiff(data):
    x = np.zeros(len(data)-1)
    for i in range(len(data)-1):
        x[i] = np.log(data[i+1]) - np.log(data[i])
    return x

def make_linear(logDiff):
    mean = np.mean(logDiff)
    x = np.zeros(len(logDiff))
    for i in range(len(logDiff)):
        x[i] = np.log((logDiff[i] - mean) ** 2)
    return x

def plots_and_prints(results, logDiff, superhelp):
    fig_14_5_i(logDiff)
    fig_14_5_ii_p1(superhelp)

    # ec)
    para_est = results.x
    LL = results.fun
    print("Parameters estimates: \n", para_est)
    print('log likelihood value: \n', LL)
    print('exit flag: \n', results.success)

def smoothing(para_est, x, y, I, RV):
    sig2_u = (np.pi ** 2) / 2  # u = log(eps_t**2)
    mean_u = -1.27

    H = np.ones(I) * sig2_u
    R = np.ones(I)
    Z = np.ones(I)
    T = np.ones(I) * para_est[1]
    Q = np.ones(I) * para_est[2]
    omega = para_est[0]
    beta = 0
    if len(para_est) == 4:
        beta = para_est[3]

    # Run KF for ML values, then run KS
    F, K, v, a, P, T = Kalman_filter(x, H, Z, R, mean_u, T, Q, omega, beta, RV)
    plt.plot(a)
    plt.plot(x, color='k')
    plt.show()
    L, V, N, alpha_hat, r = Kalman_state_smoothing(x, a, v, P, F, K, T, Z)
    fig_14_5_ii_p2(x, alpha_hat, a)  # For loop klein beetje aanpassen

def runSV(y, x, theta_ini, theta_bnds, options, RV=0):
    I = len(x)
    results = optimize.minimize(llik_fun_SV, theta_ini, args=(x, I, RV),
                                options=options,
                                method='SLSQP', bounds=(theta_bnds))
    plots_and_prints(results, y, x)
    para_est = results.x
    smoothing(para_est, x, y, I, RV)

def main():
    print('GBP_USD')
    SV_data = loadDatasv('sv.dat')
    GBP_USD_y = np.array(SV_data) / 100
    GBP_USD_x = make_linear(GBP_USD_y)
    theta_ini, theta_bnds, options = initializize_optimization()

    print('\t\t\t-----GBP_USD-----\n\n')
    runSV(GBP_USD_y, GBP_USD_x, theta_ini, theta_bnds, options)

    SP_500_data = loadDataOxford('oxfordmanrealizedvolatilityindices.csv', columnOfData=5)
    SP_500_y = convertToLogDiff(SP_500_data)
    SP_500_x = make_linear(SP_500_y)

    print('\t\t\t-----SP_500-----\n\n')
    runSV(SP_500_y, SP_500_x, theta_ini, theta_bnds, options)

    RV5 = loadDataOxford('oxfordmanrealizedvolatilityindices.csv', columnOfData=11)
    RV5 = list(np.log(RV5))
    theta_ini.append(1)
    theta_bnds.append((-np.inf, np.inf))

    print('\t\t\t-----SP_500 + extension-----\n\n')
    runSV(SP_500_y, SP_500_x, theta_ini, theta_bnds, options, RV=RV5)


if __name__ == '__main__':
    start_time = time.time()
    print('running...')
    main()
    total_time = time.time() - start_time
    hours = int(total_time / 3600)
    minutes = int((total_time - hours * 3600) / 60)
    seconds = int(total_time - hours * 3600 - minutes * 60)
    print('\n\n----- runtime %2.0f hour(s)\t%2.0f minute(s)\t%2.0f second(s)  -----' % (hours, minutes, seconds))
