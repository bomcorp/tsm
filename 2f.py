import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def main():
  
    global I
    np.random.seed(8698)
    
    # Import data
    file_name = 'data/sv.dat'
    columns = ['GBP/USD'] # GBP/USD daily exchange rates
    df = pd.read_csv(file_name, names = columns, header = 0) 
    df = pd.DataFrame(df)
    y = df['GBP/USD'] / 100
    I = len(y)
    mean_y = np.mean(y)
    x = np.log((y - mean_y)**2)
    
    N = 1000
    omega = -0.08774252
    phi = 0.99123019
    sig2_eta = 0.00699875
    mean_AR = 0 
    var_AR = sig2_eta/(1-phi**2)
    
    a = np.zeros((I,N))
    
    # Initialize alpha
    for i in range(N):
      a[0,i] = np.random.normal(mean_AR, var_AR, 1)
    
    for j in range(I):
      for i in range(1,N):
        # Draw N values alpha
        a[j,i] = np.random.normal(phi*a[j-1,i], sig2_eta, 1)  
    
    plt.plot(a, linestyle="",marker="o", color='k', markersize=2 )
    plt.show()
    
    w = np.zeros((I,N))
    w_sum = np.zeros(I)
    w_norm = np.zeros((I,N))
    
    for j in range(I):
      for i in range(N):
        # compute weights
        w[j,i] = np.exp(-0.5*np.log(2*np.pi*sig2_eta) - 0.5*a[j,i] - 0.5*sig2_eta*np.exp(-a[j,i])*(x[j]**2))
      w_sum[j] = np.sum(w[j,:])
    
    for j in range(I):
      for i in range(N):
        # normalize weights
        w_norm[j,i] = w[j,i] / w_sum[j]
    
    plt.plot(w, linestyle="",marker="o", color='k', markersize=2 )
    plt.show()
    
    plt.plot(w_norm, linestyle="",marker="o", color='k', markersize=2 )
    plt.show()

    a_hat = np.zeros(I)
    a_temp = np.zeros(N)
    
    for j in range(I):
      for i in range(N):
        # compute filtering expectation
        a_temp[i] = w_norm[j,i]*a[j,i]
      a_hat[j] = np.sum(a_temp)
      a_temp = np.zeros(N)
    
    plt.plot(a_hat, color='r')
    plt.show()
    
    P_hat = np.zeros(I)
    P_temp = np.zeros(N)
    
    for j in range(I):
      for i in range(N):
        # compute filtering variance
        P_temp[i] = w_norm[j,i]*(a[j,i]**2) - a_hat[j]**2
      P_hat[j] = np.sum(P_temp)
      P_temp = np.zeros(N)
    
    plt.plot(P_hat, color='r')
    plt.show()
    
    ESS = 100*(1/w_sum**2)
    print(ESS)
    
    
if __name__ == '__main__':
    main()
