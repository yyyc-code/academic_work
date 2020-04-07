# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 13:59:17 2018

@author: Yumeng Cui
"""

__author__ = "YumengCui"
__copyright__ = "Copyright 2018, YumengCui"
__license__ = "Questrom"
__version__ = "2.0.0"
__maintainer__ = "YumengCui"
__email__ = "ymcui@bu.edu"
__status__ = "Production"

from scipy.stats import norm
import math
import pandas as pd
import numpy as np


class stochastic_process(object):
    def __init__(self,*,r,s0,sigma,ttm ,deltat,N):
        self.r = r
        self.s0=s0
        self.sigma = sigma
        self.ttm =ttm
        self.deltat = deltat
        self.N = N
        
    def GBM_simulation(self,method):
        if method == 'Eular':
            st = pd.DataFrame()
            st['t=0'] = self.s0*np.ones(self.N,dtype=int)
            for i in range(0,int(1/self.deltat)):
                st['t=%i' %(i+1)] = st['t=%i' %i]+ self.r*self.deltat*st['t=%i' %i]+self.sigma*math.sqrt(self.deltat)*st['t=%i' %i]*np.random.randn(self.N+5000)[5000:]
        elif method == 'Formula':
            print('Method is not defined yet!')
        else:
            raise TypeError('Undefined Method!')     
        return st
    
    def Bachelier_simulation(self):
        st = pd.DataFrame()
        st['t=0'] = self.s0*np.ones(self.N,dtype=int)
        for i in range(0,int(1/self.deltat)):
            st['t=%i' %(i+1)] = st['t=%i' %i]+ self.r* st['t=%i' %i]*self.deltat+self.sigma*st['t=%i' %i]*math.sqrt(self.deltat)*np.random.randn(self.N+5000)[5000:]
        return st
    
    def CEV_simulation(self,beta):
        st = pd.DataFrame()
        st['t=0'] = self.s0*np.ones(self.N,dtype=int)
        for i in range(0,int(1/self.deltat )):
            st['t=%i' %(i+1)] = st['t=%i' %i]+ self.r* st['t=%i' %i]*self.deltat+self.sigma*np.power(st['t=%i' %i],beta)*math.sqrt(self.deltat)*np.random.randn(self.N+5000)[5000:]
        return st
        
        
        
class option_price(stochastic_process):
    def __init__(self,*,K,optionType,r,s0,sigma,ttm,deltat=[],N=[],model='GBM',method='Eular',BSFormula = False,beta = []):
        self.K = K
        self.optionType = optionType
        self.model = model
        self.method = method
        self.beta = beta
        super().__init__(r=r,s0=s0,sigma=sigma,ttm=ttm,deltat =deltat,N=N)
        if BSFormula:
            self.optionPrice = self.BS_formula()
        else:
            method_name = 'simulation_' + model
            method = getattr(self, method_name, lambda: "nothing")
            return method()
    
    def simulation_GBM(self):
        self.simulatedPrice = super().GBM_simulation(self.method)
        
    def simulation_Bachelier(self):
        self.simulatedPrice = super().Bachelier_simulation()

        
    def get_simulatedPath(self):
        return self.simulatedPrice
    
    def simulation_CEV(self):
        self.simulatedPrice = super().CEV_simulation(self.beta)    
    
    def simulation_price(self,exotic = 'European'):

        price = self.simulatedPrice
      
        if self.optionType == 'C' and exotic == 'European':
            payoff = price.iloc[:,-1] - self.K
            payoff[payoff<0]=0
            p = payoff.mean()*math.exp(-self.r*self.ttm)
        elif self.optionType == 'P' and exotic == 'European':
             payoff = self.K-price.iloc[:,-1]
             payoff[payoff<0]=0
             p = payoff.mean()*math.exp(-self.r*self.ttm)
        elif self.optionType == 'C' and exotic == 'Lookback':
             payoff = self.K - price.max(axis=1)
             payoff[payoff<0]=0
             p = payoff.mean()*math.exp(-self.r*self.ttm)
        elif self.optionType == 'P' and exotic == 'Lookback':
             payoff = self.K - price.min(axis = 1)
             payoff[payoff<0]=0
             p = payoff.mean()*math.exp(-self.r*self.ttm)
        else:
            raise TypeError()      
            
        return p
                
    
    
    def BS_formula(self):
        if self.optionType == 'C':
            sigmaRtT = ( self.sigma * math.sqrt(self.ttm))
            rSigTerm = ( self.r + self.sigma * self.sigma /2.0) * self.ttm
            d1 = ( math.log ( self.s0/ self.K) + rSigTerm ) / sigmaRtT
            self.delta = norm.cdf(d1,0,1)
            d2 = d1 - sigmaRtT
            term1 = self.s0 * norm.cdf (d1)
            term2 = self.K * math.exp (- self.r * self.ttm) * norm.cdf(d2)
            return term1 - term2
        elif self.optionType == 'P':
            sigmaRtT = ( self.sigma * math.sqrt(self.ttm))
            rSigTerm = ( self.r + self.sigma * self.sigma /2.0) * self.ttm
            d1 = ( math.log ( self.s0/ self.K) + rSigTerm ) / sigmaRtT
            d2 = d1 - sigmaRtT
            term2 = self.s0 * norm.cdf(-d1)
            term1 = self.K * math.exp(-self.r * self.ttm)*norm.cdf(-d2)
            return term1 - term2
        else:
            raise TypeError('Option type is not defined!')
        
    def get_delta(self):
        if self.optionType == 'C':
           if self.ttm == 0.0:
               delta = 1
           sigmaRtT = ( self.sigma * math.sqrt(self.ttm))
           rSigTerm = ( self.r + self.sigma * self.sigma /2.0) * self.ttm
           d1 = ( math.log ( self.s0/ self.K) + rSigTerm ) / sigmaRtT
           delta = norm.cdf(d1,0,1)
           return delta
       
            

      

    
 
        
    '''
    def premium_lookback(self):
        price_E = BS_(s0=s0,K=K,r=r,sigma=sigma,ttm=ttm,optionType ='P')
        temp = K - GBM_stimulation(r=r,s0=s0,sigma=sigma,ttm=ttm ,deltat=deltat,N=N).min(axis = 1)
        temp[temp<0]=0
        price_L = temp.mean()*math.exp(-r*ttm)
        return price_L-price_E
    '''