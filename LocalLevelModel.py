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
        df : pandas
            dataset
        or_df : str
            original dataset of the locallevelmodel
        sigma_e2 : int
        sigma_eta2 : int
        forecast : str
            

        Methods
        -------
        info(additional=""):
            Prints the local level models 's df.
    """


    def __init__(self, data, or_data, sigma_e2_hat,sigma_eta2_hat, forecast):
        self.df = data
        self.or_df = or_data
        self.sigma_e2 = sigma_e2_hat
        self.sigma_eta2 = sigma_eta2_hat
        self.forecast = forecast
        


    def initialize(self, A_1, P_1, startyear):
        '''adds all the variables to one matrix '''
        d = pd.DataFrame(np.zeros(((len(self.df)), 29)))
        columns = [ 'y', 'v_t', 'a_t','a_t_lower','a_t_upper','a_t_lower_5','a_t_upper_5', 'F_t', 'P_t', 'K_t', 'P_tt', 'a_tt', 'a_t1', 'P_t1','L_t','R_t', 'a_hat_t','a_hat_t_lower','a_hat_t_upper', 'N_t','V_t','D_t', 'e_t', 'e_t_std','eta_t', 'eta_t_std', 'error', 'u_t', 'u_star', 'r_star']
        self.df = pd.concat([self.df,d ], axis = 1, ignore_index = True)
        self.df.columns = columns
        self.df.at[0, 'P_t'] = P_1
        self.df.at[0, 'a_t'] = A_1
        self.startyear = int(startyear)


    def calc_moment(self,q, e, m1):
        '''calculate moment '''
        m = 0
        for x in e:
            m += (x-m1) ** q
        return m/len(e)

    def calc_c(self,j, m1, m2, e):
        c = 0
        for i in range(j, len(e)):
            c += (e[i] - m1) * (e[i-j] - m1) / (len(e) * m2)
        return c

    def calc_Q(self,e, k, m1, m2):
        result = 0
        for j in range(k):
            c_j = self.calc_c(j, m1, m2, e)
            result += len(e) * (len(e) + 2) * c_j ** 2 / (len(e) - j)
        return result

    def calc_H(self,e, h):
        numerator = 0
        for i in range(len(e)-h, len(e)):
            numerator += e[i] ** 2
        denominator = 0
        for i in range(h):
            denominator += e[i] ** 2
        return numerator / denominator

    def add_diagnostics(self, h, k):
        self.diagnostics_columns =  ['m1', 'm2', 'm3', 'm4', 'S', 'excess K', 'H(%d)' % h, 'Q(%d)' % k]
        e = self.df['error'].values.tolist()
        m1 = sum(e) / len(e)
        m2 = self.calc_moment(2, e, m1)
        m3 = self.calc_moment(3, e, m1)
        m4 = self.calc_moment(4, e, m1)
        S = m3 / np.sqrt(m2 ** 3)
        K = m4 / (m2 ** 2) - 3
        H = self.calc_H(e, h)
        Q = self.calc_Q(e, k, m1, m2)
        self.diagnostics_values = [m1, m2, m3, m4, S, K, H, Q]

    def print_diagnostics(self):
        print('_______________________________')
        print('Diagnostics')
        print('-------------------------------')
        e = self.df['error'].values.tolist()
        for i in range(len(self.diagnostics_columns)):
            print('%12s   %.3f' % (self.diagnostics_columns[i], self.diagnostics_values[i]))
        print('-------------------------------')
        print('Kurtosis & Skew')
        print(sc.stats.kurtosis(e), sc.stats.skew(e))
        print('_______________________________')
            

    def walkforward(self):
        '''calculates kalman filter '''
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
        '''calculates smoothing '''
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
            x_curr['u_t']= 1.0 / x_curr['F_t'] * x_curr['v_t'] - x_curr['K_t'] * x_curr['R_t']
            
            if(x != nf.shape[0]-1 ):
                x_curr['u_star'] = 1.0 / np.sqrt(x_curr['D_t']) * x_curr['u_t']
                x_curr['r_star'] = 1.0 / np.sqrt(x_curr['N_t']) * x_curr['R_t'] if x_curr['R_t'] != 0 else np.nan
            
        self.df = pd.DataFrame(data=nf)


    def plot(self,title, subtitle):
        fig, ax = plt.subplots(2,2,constrained_layout=True)
        fig.set_figheight(8)
        fig.set_figwidth(8)
        fig.suptitle(title, fontsize=14, fontweight='bold')
        ax[0,0].set_title(subtitle[0])
        ax[0,1].set_title(subtitle[1])
        ax[1,0].set_title(subtitle[2])
        ax[1,1].set_title(subtitle[3])
        return fig, ax
    
    def plot_legend(self, ax):
        ax[0,0].legend()
        ax[1,0].legend()
        ax[0,1].legend()
        ax[1,1].legend()
        plt.show()

    
    def plot_2_1(self):
        fig, ax = self.plot('2.1: data, filtered state and confidence bounds', [
            'data (dots), filtered state at',
            'filtered state variance Pt', 
            'prediction errors vt',
            'prediction variance Ft'
            ])
        st = self.startyear
        ax[0,0].plot(self.df.index+st, self.df['y'],'b.', label='y')
        ax[0,0].plot(self.df.index[1:]+st, self.df.loc[1:, ['a_t']],'-k', label='a_t',)
        ax[0,0].plot(self.df.index[1:]+st, self.df.loc[1:, ['a_t_lower']],'--g', label='lowerbound')
        ax[0,0].plot(self.df.index[1:]+st, self.df.loc[1:, ['a_t_upper']],'--g', label='upperbound')
        ax[0,1].plot(self.df.index[1:]+st, self.df.loc[1:, ['P_t']],'-k', label='P_t')
        ax[1,0].plot(self.df.index[1:]+st, self.df.loc[1:, ['v_t']],'-k', label='v_t')
        ax[1,0].axhline(y=0)
        ax[1,1].plot(self.df.index[1:]+st, self.df.loc[1:, ['F_t']],'-k', label='F_t')
        self.plot_legend(ax)
        #plt.savefig('plots/Fig_Nile_2.1.png')
        #plt.close()

        
    def plot_2_2(self):
        fig, ax = self.plot('2.2: output of state smoothing recursion', [
            'data (dots), smoothed state ˆαt',
            'smoothed state variance Vt', 
            'smoothing cumulant rt',
            'smoothing variance cumulant Nt'
            ])       
        st = self.startyear
        ax[0,0].plot(self.df.index+st, self.df['y'],'b.', label='y')
        ax[0,0].plot(self.df.index[1:]+st, self.df.loc[1:, ['a_hat_t']],'-k', label='a_hat_t',)
        ax[0,0].plot(self.df.index[1:]+st, self.df.loc[1:, ['a_hat_t_lower']],'--g', label='lowerbound')
        ax[0,0].plot(self.df.index[1:]+st, self.df.loc[1:, ['a_hat_t_upper']],'--g', label='upperbound')
        ax[0,1].plot(self.df.index[1:(len(self.df)-1)]+st, self.df.loc[1:(len(self.df)-2), ['V_t']],'-k', label='V_t')
        ax[1,0].plot(self.df.index[1:]+st, self.df.loc[1:, ['R_t']],'-k', label='r_t')
        ax[1,0].axhline(y=0)
        ax[1,1].plot(self.df.index[:(len(self.df)-1)]+st, self.df.loc[:(len(self.df)-2), ['N_t']],'-k', label='N_t')
        self.plot_legend(ax)
        #plt.savefig('plots/Fig_Nile_2.2.png')
        #plt.close()


    def plot_2_3(self):
        fig, ax = self.plot('2.3: output of disturbance smoothing recursion', [
            'observation error ˆεt',
            'observation error standard deviation', 
            'state error ˆηt',
            'state error standard deviation'
            ])  
        st = self.startyear 
        ax[0,0].plot(self.df.index+st, self.df.loc[:, ['e_t']],'-k', label='observation error')
        ax[0,0].axhline(y=0)
        ax[0,1].plot(self.df.index+st, self.df.loc[:, ['e_t_std']],'-k', label='std of observation error')
        ax[1,0].plot(self.df.index+st, self.df.loc[:, ['eta_t']],'-k', label='state error')
        ax[1,0].axhline(y=0)
        ax[1,1].plot(self.df.index+st, self.df.loc[:, ['eta_t_std']],'-k', label='std of state error')
        self.plot_legend(ax)
        #plt.savefig('plots/Fig_Nile_2.3.png')
        #plt.close()


    def plot_2_5(self):
        fig, ax = self.plot('2.5: output when observations are missing', [
            'data and filtered state at (extrapolation)',
            'filtered state variance Pt', 
            'data and smoothed state ˆαt (interpolation)',
            'smoothed state variance Vt'
            ])  
        st = self.startyear
        ax[0,0].plot(self.or_df.index+st, self.or_df['y'],'-k', label='y')
        ax[0,0].plot(self.df.index[1:]+st, self.df.loc[1:, ['a_t']],'-b', label='a_t')
        ax[0,1].plot(self.df.index[1:]+st, self.df.loc[1:, ['P_t']],'-k', label='P_t')
        ax[1,0].plot(self.or_df.index+st, self.or_df['y'],'-k', label='y')
        ax[1,0].plot(self.df.index[1:]+st, self.df.loc[1:, ['a_hat_t']],'-b', label='a_hat_t')
        ax[1,1].plot(self.df.index[1:(len(self.df)-1)]+st, self.df.loc[1:(len(self.df)-2), ['V_t']],'-k', label='V_t')
        self.plot_legend(ax)
        #plt.savefig('plots/Fig_Nile_2.5.png')
        #plt.close()


    def plot_2_6(self):
        fig, ax = self.plot('2.6: output of forecasting', [
            'data (dots), state forecast',
            'state variance Pt', 
            'observation forecast E(yt|Yt−1)',
            'observation forecast variance Ft'
            ])  
        st = self.startyear
        n_rows = len(self.df)-int(self.forecast)
        ax[0,0].plot(self.or_df.index+st, self.or_df['y'],'b.', label='y')
        ax[0,0].plot(self.df.index[1:]+st, self.df.loc[1:, ['a_t']],'-k', label='a_t')
        ax[0,0].plot(self.df.index[n_rows:]+st, self.df.loc[n_rows:, ['a_t_lower_5']],'--g', label='lowerbound')
        ax[0,0].plot(self.df.index[n_rows:]+st, self.df.loc[n_rows:, ['a_t_upper_5']],'--g', label='upperbound')
        ax[0,1].plot(self.df.index[1:]+st, self.df.loc[1:, ['P_t']],'-k', label='P_t')
        ax[1,0].plot(self.df.index[1:]+st, self.df.loc[1:, ['a_t']],'-k', label='a_t')
        ax[1,1].plot(self.df.index[1:]+st, self.df.loc[1:, ['F_t']],'-k', label='F_t')
        self.plot_legend(ax)
        #plt.savefig('plots/Fig_Nile_2.6.png')
        #plt.close()


    def plot_2_7(self):
        fig, ax = self.plot('2.7: output standardised prediction errors', [
            'standardised residual',
            'histogram plus estimated density', 
            'ordered residuals',
            'correlogram'
            ]) 
        st = self.startyear
        ax[0,0].plot(self.df.index+st,self.df['error'],label= 'standardised residual')
        ax[0,0].axhline(y=0)
        density = stats.gaussian_kde(self.df['error'])
        x_lim = np.linspace(-3.6,3.6)
        ax[0,1].hist(self.df['error'], bins=12, density=1, histtype='bar', ec='k', color='white')
        ax[0,1].plot(x_lim, density(x_lim), color='k')
        gofplots.ProbPlot(np.sort(self.df['error'])).qqplot(line="45", ax = ax[1,0])
        ax[1,0].axhline(y=0)
        ax[1,1].acorr(self.df['error'])
        ax[1,1].set_xlim([0, 11])
        ax[1,1].set_ylim([-0.8, 0.8])
        self.plot_legend(ax)
        #plt.savefig('plots/Fig_Nile_2.7.png')
        #plt.close()


    def plot_2_8(self):
        fig, ax = self.plot('2.8: output Diagnostic plots for auxiliary residuals', [
            'observation residual u∗t',
            'histogram and estimated density for u∗t', 
            'state residual r∗t',
            'histogram and estimated density for r∗t'
            ]) 
        st = self.startyear
        x_lim = np.linspace(-4,3)
        ax[0,0].plot(self.df.index[:(len(self.df)-1)]+st, self.df.loc[:(len(self.df)-2), ['u_star']],'-k', label='u_star')
        ax[0,0].axhline(y=0)
        density = stats.gaussian_kde(self.df['u_star'].dropna().values.tolist())
        ax[0,1].hist(self.df.loc[:(len(self.df)-2)+st, ['u_star']], histtype='bar', ec='k', color='white', density=1, bins=13)
        ax[0,1].plot(x_lim, density(x_lim), color='k')
        ax[1,0].plot(self.df.index[:(len(self.df)-1)]+st, self.df.loc[:(len(self.df)-2), ['r_star']],'-k', label='r_star')
        ax[1,0].axhline(y=0)
        density = stats.gaussian_kde(self.df['r_star'].dropna().values.tolist())
        ax[1,1].hist(self.df.loc[:(len(self.df)-2)+st, ['r_star']], histtype='bar', ec='k', color='white', density=1, bins=13)
        ax[1,1].plot(x_lim, density(x_lim), color='k')
        self.plot_legend(ax)
        #plt.savefig('plots/Fig_Nile_2.8.png')
        #plt.close()

      


