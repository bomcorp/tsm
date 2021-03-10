import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize, stats
import numdifftools as nd
import datetime
import csv
import time

def draw_numbers(mean, variance, N, seed):
    x = np.zeros(N)
    for i in range(N):
        seed += 1
        np.random.seed(seed)
        x[i] = np.random.normal(mean[i], np.sqrt(variance), 1)
    return x, seed

def calc_weights(alpha_tilde, N, y, sig2_epsilon, prev_weights):
    vector = np.zeros(N)
    for i in range(N):
        sub1 = -0.5 * np.log(2 * np.pi)
        sub2 = -0.5 * np.log(sig2_epsilon)
        sub3 = - 0.5 * (sig2_epsilon ** (-1))
        sub4 = (y - alpha_tilde[i]) ** 2
        number = prev_weights[i] * np.exp(sub1 + sub2 + sub3 * sub4) ################
        vector[i] = number
    return vector

def normalise(vector):
    total = sum(vector)
    return np.array(vector) / total

def calc_P(w_norm, alpha_tilde, a_hat, N):
    total = 0
    for i in range(N):
        total += w_norm[i] * (alpha_tilde[i] ** 2)
    return total - a_hat ** 2

def startup():
    # Import data
    columns = ["Annual flow volume at Aswan (river Nile)"]
    csv_file = 'data/SPX_2012_now.csv'
    df = pd.read_csv(csv_file, names=columns, header=0)

    ### y = data/observations
    y = df["Annual flow volume at Aswan (river Nile)"]

    # omega = -0.08774252
    # phi = 0.99123019
    # sig2_eta = 1469.1
    # sig2_epsilon = 15099

    #     Perform parameter estimation...
    # q: 593067697.554        sigma_e2_hat: 0.000     sigma_eta2_hat: 682.469
    # estimates:
    # sigma_e2_hat: 1.1507439235540047e-06
    # sigma_eta2_hat: [682.46904922]

    I = len(y)
    N = 10000
    omega = -0.08774252
    phi = 0.99123019
    sig2_eta = 1469.1
    sig2_epsilon = 15099

    alpha_tilde = np.zeros((I, N))

    w = np.zeros((I, N))
    w_norm = np.zeros((I, N))

    a_hat = np.zeros(I)

    P_hat = np.zeros(I)

    return y, I, N, sig2_eta, sig2_epsilon, alpha_tilde, w, w_norm, a_hat, P_hat

def main():
    seed = 5023
    y, I, N, sig2_eta, sig2_epsilon, alpha_tilde, w, w_norm, a_hat, P_hat = startup()

    alpha_tilde[0, :], seed = draw_numbers(np.ones(N) * np.mean(y), sig2_eta, N, seed)
    w[0, :] = calc_weights(alpha_tilde[0, :], N, y[0], sig2_epsilon, np.ones(N) / N)
    w_norm[0, :] = normalise(w[0, :])
    a_hat[0] = sum(np.array(w_norm[0, :]) * np.array(alpha_tilde[0, :]))
    P_hat[0] = calc_P(w_norm[0, :], alpha_tilde[0, :], a_hat[0], N)

    for t in range(1, I):
        alpha_tilde[t, :], seed = draw_numbers(alpha_tilde[t-1, :], sig2_eta, N, seed)
        w[t, :] = calc_weights(alpha_tilde[t, :], N, y[t], sig2_epsilon, w_norm[t-1, :])
        w_norm[t, :] = normalise(w[t, :])
        a_hat[t] = sum(np.array(w_norm[t, :]) * np.array(alpha_tilde[t, :]))
        P_hat[t] = calc_P(w_norm[t, :], alpha_tilde[t, :], a_hat[t], N)

    # plt.plot(alpha_tilde, linestyle="", marker="o", color='k', markersize=2)
    # plt.title('a')
    # plt.show()
    #
    # plt.plot(w, linestyle="", marker="o", color='k', markersize=2)
    # plt.title('w')
    # plt.show()
    #
    # plt.plot(w_norm, linestyle="", marker="o", color='k', markersize=2)
    # plt.title('w_norm')
    # plt.show()

    plt.plot(y, linestyle="", marker="o", color='k', markersize=2)
    plt.plot(a_hat, color='r')
    plt.title('a_hat')
    plt.show()

    plt.plot(P_hat, color='r')
    plt.title('P_hat')
    plt.show()

    ESS = 100 * (1 / sum(np.array(w_norm)**2))
    # print(ESS)


if __name__ == '__main__':
    start_time = time.time()
    print('running...')
    main()
    total_time = time.time() - start_time
    hours = int(total_time / 3600)
    minutes = int((total_time - hours * 3600) / 60)
    seconds = int(total_time - hours * 3600 - minutes * 60)
    print('\n\n----- runtime %2.0f hour(s)\t%2.0f minute(s)\t%2.0f second(s)  -----' % (hours, minutes, seconds))