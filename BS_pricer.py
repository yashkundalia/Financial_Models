import numpy as np
import pandas as pd
import scipy as scp
from scipy.sparse.linalg import spsolve
from scipy import sparse
from scipy.sparse.linalg import splu
import matplotlib.pyplot as plt
from matplotlib import cm
from time import time
import scipy.stats as ss
from functools import partial


class BS_pricer:

    def __init__(self, option_info, process_info):

        self.r = process_info.r #interest rate
        self.sigma = process_info.sig   
        self.S0 = option_info.S0 
        self.K = option_info.K
        self.T = option_info.T #in years
        self.exp_rv = process_info.exp_RV  #for GBM
        self.exercise = option_info.exercise
        self.payoff = option_info.payoff


    def payoff_f(self,S):

        if self.payoff == "call" :
            return np.maximum(S - self.K , 0)
        elif self.payoff == "put":
            return np.maximum(self.K -S ,0)

    @staticmethod
    def BS_closed(payoff="call", S0=100.0, K=100.0, T=1.0, r=0.1, sigma=0.2):

        d1 = (np.log(S0/K) + (r + (sigma**2)/2)*T)/(sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        if payoff == "call" :
            price = S0*(ss.norm.cdf(d1)) - K*np.exp(-1*r*T)*(ss.norm.cdf(d2))
                    
        elif payoff =="put" :                
            price = K*np.exp(-1*r*T)*(ss.norm.cdf(-1*d2)) - S0*(ss.norm.cdf(-1*d1))

        else:
            raise ValueError("Only 'call' or 'put' payoffs valid")

        return price

    @staticmethod
    def BS_vega(sigma , S0,K,T,r):
        """calculates change in option price with respect to volatility"""
        d1 = (np.log(S0/K) + (r + (sigma**2)/2)*T)/(sigma*np.sqrt(T))
        return S0*np.sqrt(T)*ss.norm.pdf(d1)


    def BS_MC(self,N,Err=False,Time=False):
        """BS Monte Carlo"""

        t_init = time()
        S_T = self.exp_rv(self.S0 ,self.T ,N)
        Payoff = self.payoff_f(S_T)
        V = scp.mean(np.exp(-self.r*self.T)*Payoff,axis=0)
        elapsed = time()-t_init
        if Err is True:
            if Time is True:
                
                return V,ss.sem(np.exp(-self.r*self.T)*Payoff) , elapsed
            else:
                return V,ss.sem(np.exp(-self.r*self.T)*Payoff)
        else:
            if Time is True:
                return V, elapsed
            else:
                return V

    
                                                     
        




