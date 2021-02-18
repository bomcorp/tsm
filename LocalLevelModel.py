import pandas as pd
import numpy as np
import math
import scipy as sc
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.graphics.gofplots as gofplots

class LocalLevelModel:
    """
        A class to represent a local level model 

        ...

        Attributes
        ----------
        name : str
            first name of the person
        surname : str
            family name of the person
        age : int
            age of the person

        Methods
        -------
        info(additional=""):
            Prints the person's name and age.
    """


    def __init__(self, data, or_data, sigma_e2_hat,sigma_eta2_hat, forecast):
        self.df = data
        self.or_df = or_data
        self.sigma_e2 = sigma_e2_hat
        self.sigma_eta2 = sigma_eta2_hat
        self.forecast = forecast


    def initialize(self, A_1, P_1):
        d = pd.DataFrame(np.zeros(((len(self.df)), 26)))
        columns = [ 'y', 'v_t', 'a_t','a_t_lower','a_t_upper','a_t_lower_5','a_t_upper_5', 'F_t', 'P_t', 'K_t', 'P_tt', 'a_tt', 'a_t1', 'P_t1','L_t','R_t', 'a_hat_t','a_hat_t_lower','a_hat_t_upper', 'N_t','V_t','D_t', 'e_t', 'e_t_std','eta_t', 'eta_t_std', 'error']
        self.df = pd.concat([self.df,d ], axis = 1, ignore_index = True)
        self.df.columns = columns
        self.df.at[0, 'P_t'] = P_1
        self.df.at[0, 'a_t'] = A_1

    def walkforward(self):
        nf = self.df.to_records(index=False)
        for x in range(0,nf.shape[0]):
            x_curr = nf[x]
            
            if(x != 0):
                x_curr['P_t'] = x_curr['P_t1']
                x_curr['a_t'] = x_curr['a_t1']
                x_curr['a_t_lower'] = x_curr['a_t'] + sc.stats.norm.ppf((1-0.9)/2) * np.sqrt(x_curr['P_t'])
                x_curr['a_t_upper'] = x_curr['a_t'] + sc.stats.norm.ppf((1+0.9)/2) * np.sqrt(x_curr['P_t'])
                x_curr['a_t_lower_5'] = x_curr['a_t'] + sc.stats.norm.ppf((1-0.5)/2) * np.sqrt(x_curr['P_t'])
                x_curr['a_t_upper_5'] = x_curr['a_t'] + sc.stats.norm.ppf((1+0.5)/2) * np.sqrt(x_curr['P_t'])
            
            x_curr['F_t'] = x_curr['P_t'] + self.sigma_e2

            if(np.isnan(x_curr['y'])):
                x_curr['K_t'] = 0.0
                x_curr['y'] = x_curr['a_t'] 
            else:
                x_curr['K_t'] = x_curr['P_t'] / x_curr['F_t']
                

            x_curr['P_tt'] = x_curr['P_t'] * (1-x_curr['K_t'])
            x_curr['v_t'] = x_curr['y'] - x_curr['a_t']
            x_curr['a_tt'] = x_curr['a_t'] + (x_curr['P_t'] / (x_curr['P_t'] + self.sigma_e2)) * x_curr['v_t']

            #next t+1
            if(x != nf.shape[0]-1 ):
                x_next = nf[x+1]
                x_next['a_t1'] = x_curr['a_tt']
                x_next['P_t1'] = x_curr['P_tt'] + self.sigma_eta2

        self.df = pd.DataFrame(data=nf)


    def walkbackward(self):
        nf = self.df.to_records(index=False)

        for x in range(nf.shape[0]-1,0, -1):
            x_curr = nf[x]

            x_curr['L_t'] = 1.0-x_curr['K_t']

            if(x != 0):
                x_last = nf[x-1]
                x_last['N_t']=x_curr['K_t']/x_curr['P_t'] + (x_curr['L_t']**2) *x_curr['N_t']
                x_last['R_t'] = x_curr['v_t']/x_curr['F_t'] + x_curr['L_t']*x_curr['R_t']
                x_curr['a_hat_t'] = x_curr['a_t']+x_curr['P_t']*x_last['R_t']
                x_curr['V_t'] = x_curr['P_t']-(x_curr['P_t']**2) * x_last['N_t']
                x_curr['a_hat_t_lower'] = x_curr['a_hat_t'] + sc.stats.norm.ppf((1-0.9)/2) * np.sqrt(x_curr['V_t'])
                x_curr['a_hat_t_upper'] = x_curr['a_hat_t'] + sc.stats.norm.ppf((1+0.9)/2) * np.sqrt(x_curr['V_t'])


        self.df = pd.DataFrame(data=nf)


    def disturbance(self):
        nf = self.df.to_records(index=False)

        for x in range(0,nf.shape[0]):
            x_curr = nf[x]
            x_curr['e_t'] = self.sigma_e2 * ((1 / x_curr['F_t']) * x_curr['v_t'] - x_curr['K_t'] * x_curr['R_t'])
            x_curr['eta_t'] = self.sigma_eta2 * x_curr['R_t']
            x_curr['eta_t_std'] = np.sqrt(self.sigma_eta2 - self.sigma_eta2 ** 2 * x_curr['N_t'])
            x_curr['D_t'] =  1.0 / x_curr['F_t'] + x_curr['K_t'] ** 2 * x_curr['N_t']
            x_curr['e_t_std'] = np.sqrt(self.sigma_e2 - self.sigma_e2 ** 2 * x_curr['D_t'])    
            x_curr['error'] = x_curr['v_t'] / np.sqrt(x_curr['F_t'])
            
        self.df = pd.DataFrame(data=nf)


    def plot(self,title):
        fig, ax = plt.subplots(2,2,constrained_layout=True)
        fig.suptitle(title, fontsize=14, fontweight='bold')
        return fig, ax
    
    def plot_legend(self,ax):
        ax[0,0].legend()
        ax[1,0].legend()
        ax[0,1].legend()
        ax[1,1].legend()
        plt.show()

    
    def plot_2_1(self):
        fig, ax = self.plot('2.1: data, smoothed state and confidence bounds')
        ax[0,0].plot(self.df.index, self.df['y'],'b.', label='y')
        ax[0,0].plot(self.df.index[1:], self.df.loc[1:, ['a_t']],'-k', label='a_t',)
        ax[0,0].plot(self.df.index[1:], self.df.loc[1:, ['a_t_lower']],'--g', label='lowerbound')
        ax[0,0].plot(self.df.index[1:], self.df.loc[1:, ['a_t_upper']],'--g', label='upperbound')
        ax[0,1].plot(self.df.index[1:], self.df.loc[1:, ['P_t']],'-k', label='P_t')
        ax[1,0].plot(self.df.index[1:], self.df.loc[1:, ['v_t']],'-k', label='v_t')
        ax[1,1].plot(self.df.index[1:], self.df.loc[1:, ['F_t']],'-k', label='F_t')
        self.plot_legend(ax)
        
    def plot_2_2(self):
        fig, ax = self.plot('2.2: output of state smoothing recursion')          
        ax[0,0].plot(self.df.index, self.df['y'],'b.', label='y')
        ax[0,0].plot(self.df.index[1:], self.df.loc[1:, ['a_hat_t']],'-k', label='a_hat_t',)
        ax[0,0].plot(self.df.index[1:], self.df.loc[1:, ['a_hat_t_lower']],'--g', label='lowerbound')
        ax[0,0].plot(self.df.index[1:], self.df.loc[1:, ['a_hat_t_upper']],'--g', label='upperbound')
        ax[0,1].plot(self.df.index[1:(len(self.df)-1)], self.df.loc[1:(len(self.df)-2), ['V_t']],'-k', label='V_t')
        ax[1,0].plot(self.df.index[1:], self.df.loc[1:, ['R_t']],'-k', label='r_t')
        ax[1,1].plot(self.df.index[:(len(self.df)-1)], self.df.loc[:(len(self.df)-2), ['N_t']],'-k', label='N_t')
        self.plot_legend(ax)

    def plot_2_3(self):
        fig, ax = self.plot('2.3: output of disturbance smoothing recursion')
        ax[0,0].plot(self.df.index, self.df.loc[:, ['e_t']],'-k', label='observation error')
        ax[0,1].plot(self.df.index, self.df.loc[:, ['e_t_std']],'-k', label='std of observation error')
        ax[1,0].plot(self.df.index, self.df.loc[:, ['eta_t']],'-k', label='state error')
        ax[1,1].plot(self.df.index, self.df.loc[:, ['eta_t_std']],'-k', label='std of state error')
        self.plot_legend(ax)

    def plot_2_5(self):
        fig, ax = self.plot('2.5: output when observations are missing')
        ax[0,0].plot(self.or_df.index, self.or_df['y'],'-k', label='y')
        ax[0,0].plot(self.df.index[1:], self.df.loc[1:, ['a_t']],'-k', label='a_t',)
        ax[0,1].plot(self.df.index[1:], self.df.loc[1:, ['P_t']],'-k', label='P_t')
        ax[1,0].plot(self.or_df.index, self.or_df['y'],'-k', label='y')
        ax[1,0].plot(self.df.index[1:], self.df.loc[1:, ['a_hat_t']],'-k', label='a_hat_t')
        ax[1,1].plot(self.df.index[1:(len(self.df)-1)], self.df.loc[1:(len(self.df)-2), ['V_t']],'-k', label='V_t')
        self.plot_legend(ax)


    def plot_2_6(self):
        fig, ax = self.plot('2.6: output of forecasting')
        n_rows = len(self.df)-int(self.forecast)
        ax[0,0].plot(self.or_df.index, self.or_df['y'],'b.', label='y')
        ax[0,0].plot(self.df.index[1:], self.df.loc[1:, ['a_t']],'-k', label='a_t')
        ax[0,0].plot(self.df.index[n_rows:], self.df.loc[n_rows:, ['a_t_lower_5']],'--g', label='lowerbound')
        ax[0,0].plot(self.df.index[n_rows:], self.df.loc[n_rows:, ['a_t_upper_5']],'--g', label='upperbound')
        ax[0,1].plot(self.df.index[1:], self.df.loc[1:, ['P_t']],'-k', label='P_t')
        ax[1,0].plot(self.df.index[1:], self.df.loc[1:, ['a_t']],'-k', label='a_t')
        ax[1,1].plot(self.df.index[1:], self.df.loc[1:, ['F_t']],'-k', label='F_t')
        self.plot_legend(ax)

    def plot_2_7(self):
        fig, ax = self.plot('2.7: output standardised residual')
        ax[0,0].plot(self.df['error'],label= 'standardised residual')
        density = stats.gaussian_kde(self.df['error'])
        x_lim = np.linspace(-3.6,3.6)
        ax[0,1].hist(self.df['error'], bins=12, density=1, histtype='bar', ec='k', color='white')
        ax[0,1].plot(x_lim, density(x_lim), color='k')
        gofplots.ProbPlot(np.sort(self.df['error'])).qqplot(line="45", ax = ax[1,0])
        ax[1,1].acorr(self.df['error'])
        ax[1,1].set_xlim([0, 11])
        ax[1,1].set_ylim([-0.8, 0.8])
        self.plot_legend(ax)

        # def plot2_7(self, name_dataset):
        # text = 'for the %s data set' % name_dataset
        # plot(self.e, 'standardised residual %s' % text, '2_7', xaxis=True)
        # density = stats.gaussian_kde(self.e)
        # x_lim = np.linspace(-3.6,3.6)
        # plt.figure()
        # plt.title('histogram of standardised residuals %s' % text)
        # plt.hist(self.e, bins=12, density=1, histtype='bar', ec='k', color='white')
        # plt.plot(x_lim, density(x_lim), color='k')
        # plt.savefig('2_7 histogram of standardised residuals %s' % text)
        # plt.close()

        # plt.figure()
        # plt.title('ordered residuals %s' % text)
        # gofplots.ProbPlot(np.sort(self.e)).qqplot(line="45")
        # plt.savefig('2_7 ordered residuals %s' % text)
        # plt.close()

        # plt.figure()
        # plt.title('correlogram of the standardised residuals %s' % text)
        # plt.acorr(self.e)
        # plt.xlim([0, 11])
        # plt.ylim([-0.8, 0.8])
        # plt.savefig('2_7 correlogram of the standardised residuals %s' % text)
        # plt.show()
        # plt.close()
        