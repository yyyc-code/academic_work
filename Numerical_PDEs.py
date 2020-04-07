# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 20:16:01 2019

@author: mdejg
"""

import numpy as np
from scipy.stats import norm
import pandas as pd

def euro_vanilla(S, K, T, r, sigma, option = 'call'):
    
    #S: spot price
    #K: strike price
    #T: time to maturity
    #r: interest rate
    #sigma: volatility of underlying asset
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    
    if option == 'call':
        result = (S * norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * norm.cdf(d2, 0.0, 1.0))
    if option == 'put':
        result = (K * np.exp(-r * T) * norm.cdf(-d2, 0.0, 1.0) - S * norm.cdf(-d1, 0.0, 1.0))
        
    return result

#set parameters
S = 276.11
K = 285
T = (pd.to_datetime('2019-09-30') - pd.to_datetime('2019-03-06')).days/365
r = 2.74/100
sigma = 15.74/100

c1 = euro_vanilla(S, K, T, r, sigma, option = 'call')
 

def optionPrice_pde(S, K, T, r, sigma, *,Ns = 250,Nt = 1000,sm = 500,K2 = None,form = 'European',option = 'call'):
    hs = sm/Ns
    ht = T/Nt
    
    s = np.arange(0, sm, hs)
    a = 1-(sigma**2)*(s**2)*ht/(hs**2)-r*ht
    l = 0.5*(sigma**2)*(s**2)*ht/(hs**2)-r*s*ht/(2*hs)
    u = 0.5*(sigma**2)*(s**2)*ht/(hs**2)+r*s*ht/(2*hs)
    A = np.diag(a[:-1])
    
    for i in range(Ns-2):
        A[i,i+1] = u[i]
        
    for i in range(Ns-2):
        A[i+1,i] = l[i+1]
        
    assert max(abs(np.linalg.eigvals(A))) <1, "the power of A goes exponentially!"

    
    if option == 'call':
        payoff = s-K
        payoff[payoff<0]=0
        c = payoff[:-1]
    
        for i in range(Nt):
            c = A.dot(c)
            c[-1] = c[-1]+u[Ns-2]*(sm-K*np.exp(-r*i*ht))
            if form == 'american': 
                c = [max(x,y) for x,y in zip(c,payoff[:-1])]
        
    elif option == 'callspread':
        
        long  = s-K
        long[long<0]=0
        short = s-K2
        short[short<0]=0
        payoff = long -short
        c = payoff[:-1]
        
        for i in range(Nt):
            c = A.dot(c)
            c[-1] = c[-1]+u[Ns-2]*(sm-K*np.exp(-r*i*ht))
            if form == 'american':
                c = [max(x,y) for x,y in zip(c,payoff[:-1])]
        
    else:
        raise TypeError('method undefined!')
        
    price =  np.interp(S, s[:-1], c)
    
    return price


cE = optionPrice_pde(S, K, T, r, sigma,Ns = 500,Nt = 10000,sm = 1000,K2 = None,form = 'European',option = 'call')
#cA = optionPrice_pde(S, K, T, r, sigma,Ns = 250,Nt = 1000,sm = 500,K2 = None,form = 'American',option = 'call')
(abs(cE-c1)*100/c1).round(4)

cSpread_E = optionPrice_pde(S, K, T, r, sigma,Ns = 400,Nt = 5000,sm = 800,K2 = 290,form = 'European',option = 'callspread')
cSpread_A = optionPrice_pde(S, K, T, r, sigma,Ns = 400,Nt = 5000,sm = 800,K2 = 290,form = 'American',option = 'callspread')

temp = euro_vanilla(S, K, T, r, sigma, option = 'call') - euro_vanilla(S, 290, T, r, sigma, option = 'call')
