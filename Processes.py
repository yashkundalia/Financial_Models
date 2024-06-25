import numpy as np
import pandas as pd
import scipy.stats as ss

class Diffusion_process:

    def __init__(self , r = 0.1,sig = 0.2,mu = 0.1):

        self.r = r
        self.mu = mu
        if(sig <= 0):
            raise ValueError("sigma must be positive")
        else :
            self.sig = sig

    def exp_RV(self,S0,T,N):

        X = ss.norm.rvs((self.r - 0.5*self.sig**2)*T , np.sqrt(T)*self.sig , N)
        S_T = S0*np.exp(X)
        return S_T.reshape((N,1))
    