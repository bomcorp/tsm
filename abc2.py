import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize, stats
import numdifftools as nd
import datetime


def initializize_optimization():
    
    # Initialize parameters of h_{t+1} 
    omega_ini = 1
    phi_ini = 0.9
    sig2_eta_ini  = 1
    
    theta_ini = [omega_ini, phi_ini, sig2_eta_ini]
    
    # Set bounds for parameters
    omega_bnds   = (-np.inf, np.inf)
    phi_bnds   = (0, 0.9999)    
    # if ub is set to 1, phi might become 1 and initialization of a in KF will be -inf
    sig2_eta_bnds    = (0, np.inf)
    
    theta_bnds = [omega_bnds, phi_bnds, sig2_eta_bnds]
    
    options = {'eps':1e-09,
         'disp': True,
         'maxiter':200}
    
    return theta_ini, theta_bnds, options
  
def Kalman_filter(y, H, Z, R, mean_u, T, Q, omega):
  
    a = np.zeros(I)
    a[0] = omega/(1-T[0])     # unconditional expectation AR(1) model

    P = np.zeros(I)
    P[0] = Q[0] / (1-T[0]**2) # unconditional variance AR(1) model
    
    v = np.zeros(I)
    F = np.zeros(I)
    K = np.zeros(I)
    
    for i in range(I):
        v[i] = y[i] - Z[i] * a[i] - mean_u # hij valt hierover
        F[i] = Z[i] * P[i] * np.transpose(Z[i]) + H[i]
        K[i] = T[i] * P[i] * np.transpose(Z[i]) * F[i]**(-1)
        if i < I-1:
            a[i+1] = T[i] * a[i] + K[i] * v[i] + omega
            P[i+1] = T[i] * P[i] * np.transpose(T[i]) + R[i] * Q[i] * np.transpose(R[i]) - K[i] * F[i] * np.transpose(K[i])
    return F, K, v, a[:-1], P[:-1], T
  
  
def calc_r(v, F, L):
  
    r = np.zeros(I)
    
    for i in range(I - 1, 1, -1):     
        r[i-1] = v[i] / F[i] + L[i] * r[i]
        
    return r
  
  
def calc_alpha_hat(r, a, P):
  
    alpha_hat = np.zeros(I - 1)
  
    for i in range(I - 2, 0, -1):   
        alpha_hat[i] = a[i] + P[i] * r[i-1]
        
    return alpha_hat 
  

def calc_N(F, L):

    N = np.zeros(I)
    
    for i in range(I - 1, 1, -1):
        N[i-1] = F[i]**(-1) + L[i]**2 * N[i]
        
    return N
  
  
def calc_V(N, P):
  
    V = np.zeros(I - 1)
 
    for i in range(I - 2, 0, -1):
        V[i] = P[i] - P[i]**2 * N[i-1]
        
    return V
  
  
def Kalman_state_smoothing(y, a, v, P, F, K, T, Z):

    L = T - K* Z
    
    r = calc_r(v, F, L)
    alpha_hat = calc_alpha_hat(r, a, P)
    N = calc_N(F, L)
    V = calc_V(N, P)

    return L, V, N, alpha_hat[1:], r
  
  
def plot_size(y, x):
    
    figure_dim = plt.rcParams["figure.figsize"]
    figure_dim[0] = y
    figure_dim[1] = x
    plt.rcParams["figure.figsize"] = figure_dim
    
    return figure_dim
  
  
def fig_14_5_i(y):
  
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


def fig_14_5_ii_p2(x, alpha_hat_adj):
  
    plt.plot(x, linestyle="", marker="o", color='k', markersize=2)
    plt.plot(alpha_hat_adj, color='grey')
    plt.title('smoothed estimate')
    figure_dim = plot_size(10, 4)
    plt.show()


def llik_fun_SV(theta_ini, y, I): 
  
    omega = theta_ini[0]
    phi = theta_ini[1]
    sig2_eta  = theta_ini[2]
    
    sig2_u = (np.pi**2)/2      # u = log(eps_t**2)
    mean_u = -1.27
    
    Q = np.ones(I) * sig2_eta
    T = np.ones(I) * phi
    Z = np.ones(I)
    H = np.ones(I) * sig2_u
    R = np.ones(I) 
    
    F, K, v, a, P, T = Kalman_filter(y, H, Z, R, mean_u, T, Q, omega)

    l = -(I/2)*np.log(2*np.pi) -(1/2)*np.sum(np.log(F) + v**2/F)
    
    # Check for negative F
    for i in range(len(F)-1):
      if F[i]<=0:
        print('Negative prediction error variance')
        
    return np.mean(l)


def calculateSV():
    global I
    # Import data
    file_name = 'data/sv.dat'
    columns = ['GBP/USD'] # GBP/USD daily exchange rates
    df = pd.read_csv(file_name, names = columns, header = 0) 
    df = pd.DataFrame(df)
    y = df['GBP/USD'] / 100
    I = len(y)
    mean_y = np.mean(y)
    x = np.log((y - mean_y)**2)
    
    theta_ini, theta_bnds, options = initializize_optimization()
    # Optimize
    results = optimize.minimize(llik_fun_SV, theta_ini, args=(x,I),
                                  options = options,
                                  method='SLSQP', bounds=(theta_bnds))
                      
    # Answers to Q's assignment 2
    
    # 2a)
    fig_14_5_i(y)   

    # 2b)
    fig_14_5_ii_p1(x)

    # 2c)
    para_est = results.x
    LL = results.fun
    print("Parameters estimates: \n", para_est) 
    print('log likelihood value: \n', LL)
    print('exit flag: \n', results.success)
    
    # hessian = nd.Hessian(llik_fun_SV)(results.x,x,I)

    # print('Standard Errors:')
    # standard_errors = np.sqrt(np.linalg.inv(hessian).diagonal())
    # print(standard_errors)
    
    # print('t-statistics:')
    # t_stats = results.x/standard_errors
    # print(t_stats)
    
    # print('p-values:')
    # print(2*(stats.norm.cdf(-np.abs(t_stats))))
    
    # 2d)
    sig2_u = (np.pi**2)/2      # u = log(eps_t**2)
    mean_u = -1.27
    
    H = np.ones(I)* sig2_u
    R = np.ones(I)
    Z = np.ones(I)
    T = np.ones(I)*para_est[1]
    Q = np.ones(I)*para_est[2]
    omega = para_est[0]
    
    # Run KF for ML values, then run KS
    F, K, v, a, P, T = Kalman_filter(x, H, Z, R, mean_u, T, Q, omega)
    L, V, N, alpha_hat, r = Kalman_state_smoothing(y, a, v, P, F, K, T, Z)
    
    kappa = 0
    alpha_hat_adj = alpha_hat  + kappa
    fig_14_5_ii_p2(x, alpha_hat_adj) # For loop klein beetje aanpassen




def calculateVI():

    global I
    # Import data
    file_name = 'data/oxfordmanrealizedvolatilityindices.csv'
    columns = ['close_price', 'rv5'] # GBP/USD daily exchange rates
    df = pd.read_csv(file_name, delimiter= ',') 
    df = pd.DataFrame(df)
    df = df[df['Symbol']=='.SPX']
    df =df.loc[df["Date"].between('2016-03-4', '2025-02-05')]
    df = df[columns]
    print(df)

    #print(df)
    #y =  df['close_price'].pct_change()
    
    y = np.log(df['close_price']).diff() #als value 0 is dan demeaning doen
    
    y.at[0] = np.average(y.dropna()) #mogen we de eerste nan wegdoen? of vullen met mean?
    print(len(y))

    
    I = len(y)
    mean_y = np.mean(y)
    #print(mean_y)
    x = np.log((y - mean_y)**2)
    y = y.values.tolist()
    for i in range(len(y)-1):
        if((y[i] - mean_y)**2<= 0.0000):
            print((y[i] - mean_y)**2)
    
    theta_ini, theta_bnds, options = initializize_optimization()
    # Optimize
    results = optimize.minimize(llik_fun_SV, theta_ini, args=(x,I),
                                  options = options,
                                  method='L-BFGS-B', bounds=(theta_bnds))



    fig_14_5_i(y)   

    # eb)
    fig_14_5_ii_p1(x)

    # ec)
    para_est = results.x
    LL = results.fun
    print("Parameters estimates: \n", para_est) 
    print('log likelihood value: \n', LL)
    print('exit flag: \n', results.success)
    
    # hessian = nd.Hessian(llik_fun_SV)(results.x,x,I)

    # print('Standard Errors:')
    # standard_errors = np.sqrt(np.linalg.inv(hessian).diagonal())
    # print(standard_errors)
    
    # print('t-statistics:')
    # t_stats = results.x/standard_errors
    # print(t_stats)
    
    # print('p-values:')
    # print(2*(stats.norm.cdf(-np.abs(t_stats))))
    
    # 2d)
    sig2_u = (np.pi**2)/2      # u = log(eps_t**2)
    mean_u = -1.27
    
    H = np.ones(I)* sig2_u
    R = np.ones(I)
    Z = np.ones(I)
    T = np.ones(I)*para_est[1]
    Q = np.ones(I)*para_est[2]
    omega = para_est[0]
    
    # Run KF for ML values, then run KS
    F, K, v, a, P, T = Kalman_filter(x, H, Z, R, mean_u, T, Q, omega)
    L, V, N, alpha_hat, r = Kalman_state_smoothing(y, a, v, P, F, K, T, Z)
    
    kappa = 0
    alpha_hat_adj = alpha_hat  + kappa
    fig_14_5_ii_p2(x, alpha_hat_adj) # For loop klein beetje aanpassen


def main():
  
    
    
    calculateVI()
    
    
if __name__ == '__main__':
    main()
