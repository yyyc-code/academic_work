# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 10:04:28 2019

@author: mdejg
"""


from option_price_script import *
import numpy as np

def deltaHedging_payoff(price,K,N=10000):
    payoff_po = price - K
    payoff_po[payoff_po<0] = 0
    payoff_ne = price - K
    payoff_ne[payoff_ne>0] = 0
    payoff = (0.45*payoff_po.sum()-0.55*payoff_ne.sum())/N
    return payoff

t = np.arange(0,253)/252
r = 0.0
s0 = 100
sigma = 0.25
option = option_price(K = 100,optionType = 'C',r = r,s0 = s0,sigma = sigma,ttm = 1,deltat= 1/252,N = 10000,model= 'CEV',beta = 1)    
price = option.simulatedPrice.iloc[:,-1]
payoff  = deltaHedging_payoff(price,100)
price1 = option.CEV_simulation(0.5).iloc[:,-1]
payoff1  = deltaHedging_payoff(price1,100)
option.sigma = 0.4
price2 = option.GBM_simulation('Eular').iloc[:,-1]
payoff2  = deltaHedging_payoff(price2,100)