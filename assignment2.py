import sys
import time
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
import csv
import statsmodels.graphics.gofplots as gofplots
import statsmodels.formula.api as smf
import statsmodels.stats.stattools as st
import json
from multiprocessing import Process, Pool
from StochasticVolatilityModel import StochasticVolatilityModel as svm
import argparse
import math


# read data ---------------------------------

def load_data(file_name, header_rows = 1, columns =0, delimiter=','):
    '''
        Returns loaded data from a file

            Parameters:
                    fileName (string): path to file (relative)
                    header_rows (int): which row is the header
                    columns (int): columns of data
                    delimiter (string): delimiter of file
            Returns:
                    dataframe with the data
    '''
    data = pd.read_csv(file_name, skiprows=0,delimiter = delimiter,skip_blank_lines=True,na_filter = True, usecols = columns,index_col=False)
    data.columns = ['y']
    
    return data

def export_data(dataset, path):
    '''writes data from dataframe to path '''
    df = pd.DataFrame(dataset)
    df.to_csv(path.replace('.', 'export.'))

def convert_data_tolist(dataset):
    '''converts dataset column to list '''
    data = dataset['y'].values.tolist()
    return data

def delete_data(dataset, missing_indices):
    '''delete ranges in the dataset '''
    for indice in missing_indices:
        dataset.loc[indice[0]:indice[1], ['y']] = np.NaN
    return dataset

def add_data(dataset, forecast):
    '''adds forecast range to dataset '''
    n_rows = len(dataset)
    for x in range(0,forecast):
        dataset.loc[(x+n_rows), ['y']] = np.NaN
    return(dataset)


# end read data ---------------------------------

# parameter estimation ---------------------------------
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

def parameter_estimation(data):
    psi = 0
    data = convert_data_tolist(data)
    x = sc.optimize.minimize(param_likelihood_func, psi, args=(data), method='BFGS')
    #print(x)
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



# end parameter estimation ---------------------------------





def main(gethelp, path='path', columns = [1], estimate='yes', missing='', forecast = '', startyear = '1871'):
    """Main script"""

    print("Running with parameters filepath: {}\ncolumns: {}\nestimate: {}\nmissing: {}\nforecast: {}".format(path, columns, estimate, missing, forecast, gethelp))

    #load dataset
    dataset = load_data(path, columns = columns)

    #parameters 
    A_1 = 0.0
    P_1 = 10000000.0
    SIGMA_e2 = 15099.0
    #q * sigma_e2
    SIGMA_eta2 = 1469.1

    #estimate parameters
    if(estimate == 'yes'):
        print("Perform parameter estimation...")
        q, sigma_e2_hat = parameter_estimation(dataset)
        sigma_eta2_hat = sigma_e2_hat * q
    else: 
        sigma_e2_hat = SIGMA_e2
        sigma_eta2_hat = SIGMA_eta2

    print("estimates: \nsigma_e2_hat: {}\nsigma_eta2_hat: {}".format(sigma_e2_hat, sigma_eta2_hat))

    #remove missing indices
    if(missing != ''):
        dataset = delete_data(dataset, np.array(json.loads(missing)))
    
    #add forecast
    if(forecast != ''):
        dataset = add_data(dataset, int(forecast))

    #save copy
    or_dataset = dataset

    #initialize
    model = svm(dataset,or_dataset, sigma_e2_hat, sigma_eta2_hat, forecast)
    model.initialize(A_1,P_1,startyear=startyear)

    #calculate
    model.walkforward()
    model.walkbackward()
    model.disturbance()

    export_data(model.df, path)
    

    model.plot_2_1()
    # #plot
    # if(missing == '' and forecast == ''):
    #     model.plot_2_1()
    #     model.plot_2_2()
    #     model.plot_2_3() 
    #     model.plot_2_7()
    #     model.plot_2_8()

    # if(missing != ''  and forecast == ''):
    #     model.plot_2_5()

    # if(missing == '' and forecast != ''):
    #     model.plot_2_6()



    #diagnostics
    model.add_diagnostics(33, 9)
    model.print_diagnostics()

def _cli():
    '''add arguments to commandline '''
    parser = argparse.ArgumentParser(
            description=__doc__,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            argument_default=argparse.SUPPRESS)
    parser.add_argument('-p', '--path', help="Enter relative path to data")
    parser.add_argument('-c', '--columns', nargs="*", type=int, default=[35, 40, 50, 60, 70, 80, 90], help="columns to keep")
    parser.add_argument('-e', '--estimate', help="This is the estimate argument fill in with yes or no")
    parser.add_argument('-m', '--missing', help="Enter missing ranges")
    parser.add_argument('-f', '--forecast', help="Enter forecast range")
    parser.add_argument('-s', '--startyear', help="Start year")
    parser.add_argument('-gh', '--gethelp', default=3, help="add the path with -p and add if the inital parameters should be estimated or standard should be used with -e yes/no")
    args = parser.parse_args()
    return vars(args)

if __name__ == '__main__':
    main(**_cli())