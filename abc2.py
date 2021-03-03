import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize, stats
import numdifftools as nd


def Kalman_filter(y, H, Z, R, mean_u, T, Q, omega):
    a = np.zeros(I)
    a[0] = omega/(1-T[0])

    P = np.zeros(I)
    P[0] = Q[0] / (1-T[0]**2)
    v = np.zeros(I)
    F = np.zeros(I)
    K = np.zeros(I)
    for i in range(I):
        v[i] = y[i] - Z[i] * a[i] - mean_u
        F[i] = Z[i] * P[i] * np.transpose(Z[i]) + H[i]
        K[i] = T[i] * P[i] * np.transpose(Z[i]) * F[i]**(-1)
        if i < I-1:
            a[i+1] = T[i] * a[i] + K[i] * v[i] + omega
            P[i+1] = T[i] * P[i] * np.transpose(T[i]) + R[i] * Q[i] * np.transpose(R[i]) - K[i] * F[i] * np.transpose(K[i])
    return F, K, v, a[:-1], P[:-1], T
  

def fig_14_5_i(data):
    plt.plot(data, color='grey')
    plt.plot(np.zeros(I), color='k')
    plt.show()


def fig_14_5_ii(data):
    plt.plot(data, linestyle="",marker="o", color='k', markersize=2 )
    plt.show()


def llik_fun_SV(theta_ini,y,I): 
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
    
    F, K, v, a, P, T = Kalman_filter(y, H, Z, R, mean_u, T, Q, omega);

    l = -(I/2)*np.log(2*np.pi) -(1/2)*np.sum(np.log(F) + v**2/F)
    
    #### Still gets negative F
    # for i in range(len(F)-1):
    #   if F[i]<=0:
    #     print(Z[i],P[i], F[i], v[i])
        
    return -np.mean(l)


def main():
    global I
    
    # Import data
    file_name = 'sv.dat'
    columns = ['GBP/USD'] # GBP/USD daily exchange rates
    df = pd.read_csv(file_name, names = columns, header = 0) 
    df = pd.DataFrame(df)
    y = df['GBP/USD'] / 100
    I = len(y)
    mean_y = np.mean(y)
    x = np.log((y - mean_y)**2)
    
    
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
    
    options ={'eps':1e-09,
         'disp': True,
         'maxiter':200}
    
    results = optimize.minimize(llik_fun_SV, theta_ini, args=(x,I),
                                  options = options,
                                  method='SLSQP', bounds=(theta_bnds))
                      
    
    # 2a)
    fig_14_5_i(y)   
    
    # 2b)
    fig_14_5_ii(x)  # KS needs to be added to this plot
    
    # 2c)
    print("Parameters estimates: ", results.x) 
    print('log likelihood value:')
    print(results.fun)
    print('exit flag:')
    print(results.success)
    
    # hessian = nd.Hessian(llik_fun_SV)(results.x,x,I)

    # print('Standard Errors:')
    # standard_errors = np.sqrt(np.linalg.inv(hessian).diagonal())
    # print(standard_errors)
    
    # print('t-statistics:')
    # t_stats = results.x/standard_errors
    # print(t_stats)
    
    # print('p-values:')
    # print(2*(stats.norm.cdf(-np.abs(t_stats))))

    
if __name__ == '__main__':
    main()
