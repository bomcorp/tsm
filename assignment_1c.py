import time
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.optimize import minimize
import scipy.stats
import csv
import statsmodels.graphics.gofplots as gofplots
import scipy.stats as stats

class Local_Level_Model:

    def __init__(self):
        self.a = []
        self.P = []
        self.v = []
        self.K = []
        self.L = []
        self.r = []
        self.F = []
        self.N = []
        self.V = []
        self.e = []
        self.alpha = []
        self.sigma_e2 = 0
        self.sigma_eta2 = 0
        self.epsilon = []
        self.eta = []
        self.std_eta = []
        self.std_epsilon = []

    def make_vectors(self, n):
        self.a = np.zeros(n)
        self.P = np.zeros(n)
        self.v = np.zeros(n)
        self.K = np.zeros(n)
        self.L = np.zeros(n)
        self.r = np.zeros(n)
        self.F = np.zeros(n)
        self.N = np.zeros(n)
        self.V = np.zeros(n)
        self.alpha = np.zeros(n)

    def initialise_model(self, a_ini, P_ini, sigma_e2, sigma_eta2, data):
        self.sigma_e2 = sigma_e2
        self.sigma_eta2 = sigma_eta2
        n = len(data)
        self.make_vectors(n)
        self.a[0] = a_ini
        self.P[0] = P_ini
        self.F, self.K, self.v, self.a, self.P = apply_kalman_filter(data, n, a_ini, P_ini, sigma_e2, sigma_eta2)

    def plot2_1(self, data, confidence, name_dataset):
        text = 'for the %s data set' % name_dataset
        # disregard the initialisation values of a and p to avoid them dominating the plots
        lowerbound, upperbound = calc_confidence_bounds(self.a[1:], self.P[1:], confidence)
        multiplot([data[1:], self.a[1:], lowerbound, upperbound],
                  'data, filtered state a and %2.0f%s conf bounds %s' % (100 * confidence, '%', text),
                  '2_1',
                  ['b.', '-k', '--g', '--g'],
                  ['data', 'a', 'lowerbound', 'upperbound'])
        plot(self.P[1:], 'filtered state variance P %s' % text, '2_1')
        plot(self.v[1:], 'prediction errors %s' % text, '2_1', xaxis=True)
        plot(self.F[1:], 'prediction variance %s' % text, '2_1')

    def calc_L(self):
        for i in range(len(self.a)):
            self.L[i] = 1 - self.K[i]

    def calc_r(self, r_ini):
        self.r[-1] = r_ini
        for i in range(1, len(self.r)):
            index = len(self.r) - 1 - i
            self.r[index] = self.v[index + 1] / self.F[index + 1] + self.L[index + 1] * self.r[index + 1]

    def calc_alpha(self):
        for i in range(1, len(self.a)):
            self.alpha[i] = self.a[i] + self.P[i] * self.r[i - 1]

    def calc_N(self, N_ini):
        self.N[-1] = N_ini
        for i in range(1, len(self.N)):
            index = len(self.N) - 1 - i
            self.N[index] = self.K[index + 1] / self.P[index + 1] + (self.L[index + 1] ** 2) * self.N[index + 1]

    def calc_V(self):
        self.V[0] = self.P[0] - (self.P[0] ** 2) * ( 1 / self.F[0] + self.L[0] ** 2 * self.N[0])
        for i in range(1, len(self.V)):
            self.V[i] = self.P[i] - (self.P[i] ** 2) * self.N[i - 1]

    def initialise_smoothing(self, r_ini, N_ini, data):
        self.calc_L()
        self.calc_r(r_ini)
        self.calc_alpha()
        self.calc_N(N_ini)
        self.calc_V()

    def plot2_2(self, data, confidence, name_dataset):
        text = 'for the %s data set' % name_dataset
        lowerbound, upperbound = calc_confidence_bounds(self.alpha[1:], self.V[1:], confidence)
        multiplot([data[1:], self.alpha[1:], lowerbound, upperbound],
                  'data, smoothed state alpha and %2.0f%s conf bounds %s' % (100 * confidence, '%', text),
                  '2_2',
                  ['b.', '-k', '--g', '--g'],
                  ['data', 'alpha', 'lowerbound', 'upperbound'])
        plot(self.V, 'smoothed state variance V %s' % text, '2_2')
        plot(self.r[:-1], 'smoothing cumulant r %s' % text, '2_2', xaxis=True)
        plot(self.N[:-1], 'smoothing variance cumulant %s' % text, '2_2')

    def calc_epsilon(self):
        epsilon = []
        for i in range(len(self.v)):
            epsilon.append(self.sigma_e2 * ((1 / self.F[i]) * self.v[i] - self.K[i] * self.r[i]))
        self.epsilon = epsilon

    def calc_eta(self):
        eta = []
        for r_value in self.r:
            eta.append(self.sigma_eta2 * r_value)
        self.eta = eta

    def calc_std_eta(self):
        std_eta = []
        for n in self.N:
            std_eta.append(np.sqrt(self.sigma_eta2 - self.sigma_eta2 ** 2 * n))
        self.std_eta = std_eta

    def calc_std_epsilon(self):
        std_epsilon = []
        for i in range(len(self.F)):
            D = 1 / self.F[i] + self.K[i] ** 2 * self.N[i]
            std_epsilon.append(np.sqrt(self.sigma_e2 - self.sigma_e2 ** 2 * D))
        self.std_epsilon = std_epsilon

    def disturbance_smoothing(self):
        self.calc_epsilon()
        self.calc_eta()
        self.calc_std_eta()
        self.calc_std_epsilon()

    def plot2_3(self, name_dataset):
        text = 'for the %s data set' % name_dataset
        plot(self.epsilon, 'observation error %s' % text, '2_3', xaxis=True)
        plot(self.std_epsilon, 'observation error variance %s' % text, '2_3')
        plot(self.eta, 'state error %s' % text, '2_3', xaxis=True)
        plot(self.std_eta, 'state error variance %s' % text, '2_3')

    def plot2_5(self, data, missing_observations, name_dataset):
        text = 'for the %s data set' % name_dataset
        missing_observations = np.concatenate(([[0, 0]], missing_observations))
        missing_observations = np.concatenate((missing_observations, [[len(data), len(data)]]))

        missing_observations_plot(data, self.a, 'data and state forecast a %s' % text, '2_5', missing_observations, 'a')

        plot(self.P[1:], 'state variance P %s' % text, '2_5')

        missing_observations_plot(data, self.alpha, 'data and smoothed state alpha %s' % text,
                                  '2_5', missing_observations, 'alpha')

        plot(self.V, 'smoothed state variance V %s' % text, '2_5')

    def standardise_forecast_errors(self):
        e = []
        for i in range(len(self.v)):
            e.append(self.v[i] / np.sqrt(self.F[i]))

        self.e = e

    def plot2_7(self, name_dataset):
        text = 'for the %s data set' % name_dataset
        plot(self.e, 'standardised residual %s' % text, '2_7', xaxis=True)
        density = stats.gaussian_kde(self.e)
        x_lim = np.linspace(-3.6,3.6)
        plt.figure()
        plt.title('histogram of standardised residuals %s' % text)
        plt.hist(self.e, bins=12, density=1, histtype='bar', ec='k', color='white')
        plt.plot(x_lim, density(x_lim), color='k')
        plt.savefig('2_7 histogram of standardised residuals %s' % text)
        plt.close()

        plt.figure()
        plt.title('ordered residuals %s' % text)
        gofplots.ProbPlot(np.sort(self.e)).qqplot(line="45")
        plt.savefig('2_7 ordered residuals %s' % text)
        plt.close()

        plt.figure()
        plt.title('correlogram of the standardised residuals %s' % text)
        plt.acorr(self.e)
        plt.xlim([0, 11])
        plt.ylim([-0.8, 0.8])
        plt.savefig('2_7 correlogram of the standardised residuals %s' % text)
        plt.show()
        plt.close()
        


    def plot2_8(self, name_dataset):
        text = 'for the %s data set' % name_dataset
        u = []
        u_star = []
        D = []
        r_star = []
        x_lim = np.linspace(-3.6,3.6)
        for i in range(len(self.a)):
            D.append(1 / self.F[i] + self.K[i] ** 2 * self.N[i])
            u.append(1 / self.F[i] * self.v[i] - self.K[i] * self.r[i])
            u_star.append(1 / np.sqrt(D[i]) * u[i])
            r_star.append(1 / np.sqrt(self.N[i]) * self.r[i])
        plot(u_star, 'observation residual %s' % text, '2_8', xaxis=True)

        plt.figure()
        density = stats.gaussian_kde(u_star)
        plt.title('histogram for the observation residuals %s' % text)
        plt.hist(u_star, histtype='bar', ec='k', color='white', density=1, bins=12)
        plt.plot(x_lim, density(x_lim), color='k')
        plt.savefig('2_8 histogram for the observation residuals %s' % text)
        plt.close()

        plot(r_star, 'state residual %s' % text, '2_8', xaxis=True)
        # print(len(r_star))
        plt.figure()
        density = stats.gaussian_kde(r_star[:-1])
        plt.title('histogram for the state residuals %s' % text)
        plt.hist(r_star, histtype='bar', ec='k', color='white', density=1, bins=12)
        plt.plot(x_lim, density(x_lim), color='k')
        plt.savefig('2_8 histogram for the state residuals %s' % text)
        plt.close()

def calc_moment(q, e, m1):
    m = 0
    for x in e:
        m += (x-m1) ** q
    return m/len(e)

def calc_moments(e, h, k):
    name_vector = ['m1', 'm2', 'm3', 'm4', 'S', 'excess K', 'H(%d)' % h, 'Q(%d)' % k]
    m1 = sum(e) / len(e)
    m2 = calc_moment(2, e, m1)
    m3 = calc_moment(3, e, m1)
    m4 = calc_moment(4, e, m1)
    S = m3 / np.sqrt(m2 ** 3)
    K = m4 / (m2 ** 2) - 3
    H = calc_H(e, h)
    Q = calc_Q(e, k, m1, m2)
    value_vector = [m1, m2, m3, m4, S, K, H, Q]
    for i in range(len(name_vector)):
        print('%12s   %.3f' % (name_vector[i], value_vector[i]))
    print(scipy.stats.kurtosis(e), scipy.stats.skew(e))

def calc_c(j, m1, m2, e):
    c = 0
    for i in range(j, len(e)):
        c += (e[i] - m1) * (e[i-j] - m1) / (len(e) * m2)
    return c

def calc_Q(e, k, m1, m2):
    result = 0
    for j in range(k):
        c_j = calc_c(j, m1, m2, e)
        result += len(e) * (len(e) + 2) * c_j ** 2 / (len(e) - j)
    return result

def calc_H(e, h):
    numerator = 0
    for i in range(len(e)-h, len(e)):
        numerator += e[i] ** 2
    denominator = 0
    for i in range(h):
        denominator += e[i] ** 2
    return numerator / denominator

def loadData(fileName, amtHeaderRows=1, columnOfData=0, delimiter=','):
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

def apply_kalman_filter(data, n, a_ini, P_ini, sigma_e2, sigma_eta2):
    F = np.zeros(n)
    K = np.zeros(n)
    v = np.zeros(n)
    # the recursion calculates vectors a and P to be 1 longer, this value is removed in the return statement
    a = np.zeros(n+1)
    P = np.zeros(n+1)
    a[0] = a_ini
    P[0] = P_ini
    for i in range(n):
        F[i], K[i], v[i], a[i+1], P[i+1] = single_kalman(data[i], a[i], P[i], sigma_e2, sigma_eta2)
    return F, K, v, a[:-1], P[:-1]

def single_kalman(datapoint, a_ini, P_ini, sigma_e2, sigma_eta2):
    F = P_ini + sigma_e2
    if datapoint == 'null':
        K = 0
        datapoint = a_ini
    else:
        K = P_ini / F
    v = datapoint - a_ini
    a = a_ini + K * v
    P = P_ini * (1 - K) + sigma_eta2
    return F, K, v, a, P

def missing_observations_plot(interrupted_data, data2, title, fignr, missing_observations, label):
    plt.figure()
    plt.title(title)
    for i in range(len(missing_observations) - 1):
        start, end = missing_observations[i][1], missing_observations[i + 1][0]
        plt.plot(np.arange(start, end - 1), interrupted_data[start: end - 1], '-b')
    plt.plot(np.arange(len(data2))[1:], data2[1:], '-k', label=label)
    plt.legend()
    plt.savefig('%s %s' % (fignr, title))
    plt.close()

def plot(data, title, fignr, xaxis=False):
    plt.figure()
    plt.title(title)
    plt.plot(data)
    if xaxis:
        x = np.zeros(len(data))
        plt.plot(x, '-k')
    fig_title = '%s %s' % (fignr, title)
    plt.savefig(fig_title)
    plt.close()

def multiplot(datas, title, fignr, plottype=[], labels=[]):
    plt.figure()
    plt.title(title)
    for i in range(len(datas)):
        plt.plot(datas[i], plottype[i], label=labels[i])
    plt.legend()
    fig_title = '%s %s' % (fignr, title)
    plt.savefig(fig_title)
    plt.close()

def calc_confidence_bounds(mean, variance, confidence_level):
    lowerbound = np.zeros(len(mean))
    upperbound = np.zeros(len(mean))
    for i in range(len(mean)):
        lowerbound[i] = mean[i] + scipy.stats.norm.ppf((1-confidence_level)/2) * np.sqrt(variance[i])
        upperbound[i] = mean[i] + scipy.stats.norm.ppf((1+confidence_level)/2) * np.sqrt(variance[i])
    return lowerbound, upperbound

def remove_data_points(data, missing_indices):
    new_data = []
    for i in range(len(data)):
        if missing_indices.__contains__(i):
            new_data.append('null')
        else:
            new_data.append(data[i])
    return new_data

def specialplot2_6(data, forecast_amt, a, forecast_a, forecast_F, text):
    upperbound, lowerbound = [], []
    for i in range(forecast_amt):
        upperbound.append(forecast_a[i] + 0.5 * np.sqrt(forecast_F[i]))
        lowerbound.append(forecast_a[i] - 0.5 * np.sqrt(forecast_F[i]))

    n = len(a)
    in_sample = np.arange(n)
    out_of_sample = np.arange(n, n + forecast_amt)
    total_sample = np.arange(n + forecast_amt)

    plt.figure()
    plt.title('data, state forecast a and 50%s confidence intervals %s' % ('%', text))
    plt.plot(total_sample[1:], np.concatenate((a[1:], forecast_a)), '-k', label='a')
    plt.plot(in_sample, data, 'b.', label='data')
    plt.plot(out_of_sample, upperbound, '--g', label='upperbound')
    plt.plot(out_of_sample, lowerbound, '--g', label='lowerbound')
    plt.legend()
    plt.savefig('2_6 data, state forecast a and 50%s confidence intervals %s' % ('%', text))
    plt.close()

def forecast_and_fig2_6(data, forecast_amt, name_dataset, model):
    text = 'for the %s data set' % name_dataset
    dataset = []
    for i in range(forecast_amt):
        dataset.append('null')
    forecast_F, forecast_K, forecast_v, forecast_a, forecast_P = \
        apply_kalman_filter(dataset, forecast_amt, model.a[-1], model.P[-1], model.sigma_e2, model.sigma_eta2)

    specialplot2_6(data, forecast_amt, model.a, forecast_a, forecast_F, text)
    plot(np.concatenate((model.P[1:], forecast_P)), 'state variance P %s' % text, '2_6')
    plot(np.concatenate((model.a[1:], forecast_a)), 'observation forecast %s' % text, '2_6')
    plot(np.concatenate((model.F[1:], forecast_F)), 'observation forecast variance F %s' % text, '2_6')

def LLM_missing_obs(data, q, sigma_e2, missing_observations, name_dataset):
    missing_indices = []
    for missing_set in missing_observations:
        missing_indices = np.concatenate((missing_indices, np.arange(missing_set[0] - 1, missing_set[1])))
    sigma_eta2 = q * sigma_e2
    missing_obs_model = Local_Level_Model()
    missing_obs_data = remove_data_points(data, missing_indices)
    missing_obs_model.initialise_model(0, 10 ** 7, sigma_e2, sigma_eta2, missing_obs_data)
    missing_obs_model.initialise_smoothing(0, 0, missing_obs_data)
    missing_obs_model.plot2_5(missing_obs_data, missing_observations, name_dataset)

def LLM(data, q, sigma_e2, name_dataset):
    sigma_eta2 = q * sigma_e2
    model = Local_Level_Model()

    model.initialise_model(0, 10 ** 7, sigma_e2, sigma_eta2, data)
    model.plot2_1(data, 0.9, name_dataset)
    model.initialise_smoothing(0, 0, data)
    model.plot2_2(data, 0.9, name_dataset)
    model.disturbance_smoothing()
    model.plot2_3(name_dataset)

    forecast_and_fig2_6(data, 30, name_dataset, model)
    model.standardise_forecast_errors()
    model.plot2_7(name_dataset)
    model.plot2_8(name_dataset)

    calc_moments(model.e, 33, 9)

def parameter_estimation(data):
    psi = 0
    x = minimize(param_likelihood_func, psi, args=(data), method='BFGS')
    print(x)
    q = np.exp(x.x)
    n = len(data)
    v, F = altered_kalman(data, q)
    sigma_e2_hat = sigma_e2_hat_calc(v, F, n)
    print('q: %.3f\tsigma_e2_hat: %.3f\tsigma_eta2_hat: %.3f' % (q, sigma_e2_hat, q*sigma_e2_hat))
    return q, sigma_e2_hat

# run the kalman filter with given initialisations from n=2,....,t
# note the return vector is still full length, with a 0 at t=1, for consistency
def altered_kalman(data, q):
    input_data = data[1:]
    altered_n = len(data) - 1
    a_ini = data[0]
    P_ini = 1 + q
    F_star, K, v, a, P_star = apply_kalman_filter(input_data, altered_n, a_ini, P_ini, 1, q)
    return np.concatenate(([0], v)), np.concatenate(([0], F_star))

def sigma_e2_hat_calc(v, F_star, n):
    sigma_e2_hat = 0
    for i in range(1, n):
        sigma_e2_hat += v[i] ** 2 / F_star[i]
    sigma_e2_hat = sigma_e2_hat / (n - 1)
    return sigma_e2_hat

def likelihood_value(n, F, sigma_e2_hat):
    total = - ((n - 1) / 2) * np.log(sigma_e2_hat)
    #-(n / 2) * np.log(2 * np.pi) - (n - 1) / 2
    for i in range(1, n):
        total -= 0.5 * np.log(F[i])
    return total

def param_likelihood_func(psi, data):
    n = len(data)
    q = np.exp(psi)
    v, F_star = altered_kalman(data, q)
    sigma_e2_hat = sigma_e2_hat_calc(v, F_star, n)
    total = likelihood_value(n, F_star, sigma_e2_hat)

    if sigma_e2_hat <= 0 or np.isnan(total):
        return np.inf

    return -total

def main():
    NileData = loadData('data/Nile.dat')
    q, sigma_e2_hat = parameter_estimation(NileData)
    LLM(NileData, q, sigma_e2_hat, 'Nile')
    missing_observations = [[21, 40], [61, 80]]
    LLM_missing_obs(NileData, q, sigma_e2_hat, missing_observations, 'Nile')

    # DutchGDPData = loadData('netherlands-gdp-growth-rate.csv', columnOfData=2)
    # q, sigma_e2_hat = parameter_estimation(DutchGDPData)
    # LLM(DutchGDPData, q, sigma_e2_hat)

if __name__ == '__main__':
    start_time = time.time()
    print('running...')
    main()
    total_time = time.time() - start_time
    hours = int(total_time/3600)
    minutes = int((total_time - hours * 3600) / 60)
    seconds = int(total_time - hours * 3600 - minutes * 60)
    print('\n\n----- runtime %2.0f hour(s)\t%2.0f minute(s)\t%2.0f second(s)  -----' % (hours, minutes, seconds))