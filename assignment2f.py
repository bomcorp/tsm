import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time


def main():
    global I
    np.random.seed(8698)

    # Import data
    file_name = 'sv.dat'
    columns = ['GBP/USD']  # GBP/USD daily exchange rates
    df = pd.read_csv(file_name, names=columns, header=0)
    df = pd.DataFrame(df)
    y = df['GBP/USD'] / 100
    I = len(y)
    mean_y = np.mean(y)
    x = np.log((y - mean_y) ** 2)

    N = 1000
    omega = -0.08774252
    phi = 0.99123019
    sig2_eta = 0.00699875
    mean_AR = 0
    var_AR = sig2_eta / (1 - phi ** 2)

    a = np.zeros((I, N))

    # Initialize alpha
    for i in range(N):
        a[0, i] = np.random.normal(mean_AR, np.sqrt(var_AR), 1)#############################sqrt variance

    for t in range(I):
        for i in range(1, N):
            # Draw N values alpha
            a[t, i] = np.random.normal(a[t - 1, i], np.sqrt(sig2_eta), 1)#################sqrt

    plt.plot(a, linestyle="", marker="o", color='k', markersize=2)
    plt.title('a')
    plt.show()

    w = np.zeros((I, N))
    w_sum = np.zeros(I)
    w_norm = np.zeros((I, N))

    for t in range(I):
        total = 0
        for i in range(N):
            # compute weights
            sub1 = -0.5 * np.log(2*np.pi*sig2_eta)
            sub2 = - 0.5 * a[t, i]
            sub3 = - (0.5/sig2_eta)
            sub4 = np.exp(-a[t, i])
            sub5 = y[t]**2 ### we moeten formule 1: y_t gebruiken, niet de x_t = log(y-u)**2
            number = np.exp(sub1 + sub2 + sub3*sub4*sub5) ################# 1/ sig2eta
            w[t, i] = number
            total += number
        w_sum[t] = total

    for j in range(I):
        for i in range(N):
            # normalize weights
            w_norm[j, i] = w[j, i] / w_sum[j]

    plt.plot(w, linestyle="", marker="o", color='k', markersize=2)
    plt.title('w')
    plt.show()

    plt.plot(w_norm, linestyle="", marker="o", color='k', markersize=2)
    plt.title('w_norm')
    plt.show()

    a_hat = np.zeros(I)
    a_temp = np.zeros(N)

    for j in range(I):
        for i in range(N):
            # compute filtering expectation
            a_temp[i] = w_norm[j, i] * a[j, i]
        a_hat[j] = np.sum(a_temp)
        a_temp = np.zeros(N)

    plt.plot(a_hat, color='r')
    plt.title('a_hat')
    plt.show()

    P_hat = np.zeros(I)
    P_temp = np.zeros(N)

    for j in range(I):
        for i in range(N):
            # compute filtering variance
            P_temp[i] = w_norm[j, i] * (a[j, i] ** 2) - a_hat[j] ** 2
        P_hat[j] = np.sum(P_temp)
        P_temp = np.zeros(N)

    plt.plot(P_hat, color='r')
    plt.title('P_hat')
    plt.show()

    ESS = 100 * (1 / w_sum ** 2)
    print(ESS)


if __name__ == '__main__':
    start_time = time.time()
    print('running...')
    main()
    total_time = time.time() - start_time
    hours = int(total_time / 3600)
    minutes = int((total_time - hours * 3600) / 60)
    seconds = int(total_time - hours * 3600 - minutes * 60)
    print('\n\n----- runtime %2.0f hour(s)\t%2.0f minute(s)\t%2.0f second(s)  -----' % (hours, minutes, seconds))