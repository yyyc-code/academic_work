# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 13:02:50 2020

Calibration of Heston Model

@author: mdejg
"""


from fftoptionlib import *
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy import interpolate, optimize
import pandas as pd

#set parameter
data = pd.read_excel('mf796-hw5-opt-data.xlsx')
underlying_price = 267.15
risk_free_rate = 0.015
dividend_rate = 0.0177
evaluation_date = pd.to_datetime('2019-03-05')
maturity_date = data.maturity
call_mid = (data.call_bid+data.call_ask)/2


vanilla_option = BasicOption()
(vanilla_option.set_underlying_close_price(underlying_price)
 .set_dividend(dividend_rate)
 .set_maturity_date(maturity_date[0])
 .set_evaluation_date(evaluation_date)
 .set_zero_rate(risk_free_rate))

put_call = 'call'

#para = [kappa,theta,sigma,rho,nu0]
ss = [0,9,25,9,25,44]
def costfun(param,strike = data.K,maturity = maturity_date.drop_duplicates(),market = call_mid):
    #get heston price
    fft_price = np.zeros(len(market))
    for i in range(len(maturity)):
        strike_arr = strike[ss[i]:ss[i+3]]
        vanilla_option.set_maturity_date(maturity.iloc[i])
        ft_pricer = FourierPricer(vanilla_option)
        ft_pricer.set_log_st_process(Heston(
            V0 = param[4],
                theta = param[1],
                k=param[0],
                sigma=param[2],
                rho=param[3]))
        
        ft_pricer.set_pricing_engine(FFTEngine(N=2**15, d_u=0.01, alpha=1.5, spline_order=2))
        fft_price[ss[i]:ss[i+3]] = ft_pricer.calc_price(strike_arr,put_call)
        
    cost = ((market - fft_price)**2).sum()
    return cost

x0 = [0.1,0.05,0.01,-0.4,0.05]
lb = [0.01,0,0,-1,0]
ub = [6,2,2,1,2]
bnds = optimize.Bounds(lb,ub)
temp = optimize.minimize(costfun,x0,method = 'L-BFGS-B',bounds = bnds)

call_spread = - data.call_bid + data.call_ask

def costfun_weighted(param,strike = data.K,maturity = maturity_date.drop_duplicates(),market = call_mid,weights=call_spread):
    w = 1/weights    
    fft_price = np.zeros(len(market))
    for i in range(len(maturity)):
        strike_arr = strike[ss[i]:ss[i+3]]
        vanilla_option.set_maturity_date(maturity.iloc[i])
        ft_pricer = FourierPricer(vanilla_option)
        ft_pricer.set_log_st_process(Heston(
            V0 = param[4],
                theta = param[1],
                k=param[0],
                sigma=param[2],
                rho=param[3]))
        
        ft_pricer.set_pricing_engine(FFTEngine(N=2**15, d_u=0.01, alpha=1.5, spline_order=2))
        fft_price[ss[i]:ss[i+3]] = ft_pricer.calc_price(strike_arr,put_call)
        
    cost = (w*(market - fft_price)**2).sum()
    return cost

x0 = [0.1,0.05,0.01,-0.4,0.05]
lb = [0.01,0,0,-1,0]
ub = [6,2,2,1,2]
bnds = optimize.Bounds(lb,ub)
temp = optimize.minimize(costfun_weighted,x0,method = 'L-BFGS-B',bounds = bnds)


