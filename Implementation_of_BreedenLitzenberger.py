# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 13:07:45 2020

@author: mdejg
"""

from fftoptionlib import *
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy import interpolate, optimize
import pandas as pd
#rnd stands for risk neutral density
#set parameters
optionDelta = np.array([-10,-25,-40,50,40,25,10])/100
volatility1 = np.array([32.25,24.73,20.21,18.24,15.74,13.7,11.48])/100
volatility3 = np.array([28.36,21.78,18.18,16.45,14.62,12.56,10.94])/100
s0 = 100
r = 0
d1 = np.append(norm.ppf(optionDelta[:3]+1,0,1), norm.ppf(optionDelta[3:],0,1))
t = np.array([1,3])/12

#a)
k1 = (s0*np.exp(0.5*volatility1*t[0]-volatility1*np.sqrt(t[0])*d1)).round(2)
k3 = (s0*np.exp(0.5*volatility3*t[1]-volatility3*np.sqrt(t[1])*d1)).round(2)

#b)
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


def extract_RND(K,sigma,max_k,min_k,expiry = 1/12,s0=100,r=0,step=0.2,s=2):
    x = K
    y = sigma
    tck = interpolate.splrep(x, y, s=s)
    k = np.arange(min_k, max_k + step, step)
    ynew = interpolate.splev(k, tck, der=0)
    C = euro_vanilla(s0,k,expiry,r,ynew)
    
    # Notifying vertical arbitrage opportunities
    dC = np.diff(C)
    if any(dC > 0):
        raise ValueError('Input call prices allow for arbtirage (vert. spread)') 
    # Notifying and dealing with butterfly arbitrage opportunities
    f = [] 
    f = (C[2:]+C[:-2]-2*C[1:-1])/step**2
    idxkArb = np.where(f<0)                                 
    if  len(idxkArb[0]) !=0:
      raise ValueError('Input call prices allow for arbitrage (butterfly)')
    
    return pd.DataFrame({'impliedVol':ynew[1:-1],'callPrice':C[1:-1],'phi(K)':f},index = k[1:-1].round(2))
    
#rnd for 1m
phi_k1 = extract_RND(k1,volatility1,max_k=110,min_k=86,expiry = 1/12)
plt.plot(k1,volatility1,'o', phi_k1.index.values, phi_k1['impliedVol'], '-')
plt.xlabel('Strike Prices')
plt.ylabel('Volatility')
plt.plot(phi_k1.index.values, phi_k1['phi(K)']/phi_k1['phi(K)'].sum(), '-')
plt.xlabel('Strike Prices')
plt.ylabel('density')
  
#rnd for 3m
phi_k3 = extract_RND(k3,volatility3,max_k=120,min_k=80,expiry = 3/12)
plt.plot(k3,volatility3,'o', phi_k3.index.values, phi_k3['impliedVol'], '-')
plt.xlabel('Strike Prices')
plt.ylabel('Volatility')

plt.plot(phi_k3.index.values, phi_k3['phi(K)']/phi_k3['phi(K)'].sum(), '-')
plt.xlabel('Strike Prices')
plt.ylabel('density')


def extract_RND_constVol(K,sigma,max_k,min_k,const_vol,expiry = 1/12,s0=100,r=0,step=0.2,s=2):

    k = np.arange(min_k, max_k + step, step)
    C = euro_vanilla(s0,k,expiry,r,const_vol)
    
    # Notifying vertical arbitrage opportunities
    dC = np.diff(C)
    if any(dC > 0):
        raise ValueError('Input call prices allow for arbtirage (vert. spread)') 
    # Notifying and dealing with butterfly arbitrage opportunities
    f = [] 
    f = (C[2:]+C[:-2]-2*C[1:-1])/step**2
    idxkArb = np.where(f<0)                                 
    if  len(idxkArb[0]) !=0:
      raise ValueError('Input call prices allow for arbitrage (butterfly)')
    
    return pd.DataFrame({'callPrice':C[1:-1],'phi(K)':f},index = k[1:-1].round(2)) 
#rnd for 1 month constant sigma
phi_k1_constVol = extract_RND_constVol(k1,volatility1,max_k=110,min_k=86,const_vol = 0.1824,expiry = 1/12)
plt.plot(phi_k1_constVol.index.values, phi_k1_constVol['phi(K)']/phi_k1_constVol['phi(K)'].sum(), '-')
plt.xlabel('Strike Prices')
plt.ylabel('density')

#rnd for 3 month constant sigma

phi_k3_constVol = extract_RND_constVol(k3,volatility3,max_k=120,min_k=80,const_vol = 0.1645,expiry = 3/12)
plt.plot(phi_k3_constVol.index.values, phi_k3_constVol['phi(K)']/phi_k3_constVol['phi(K)'].sum(), '-')
plt.xlabel('Strike Prices')
plt.ylabel('density')



#price digital option
phi_k1['phi(K)'].loc[:110].sum()/phi_k1['phi(K)'].sum()
1 - phi_k3['phi(K)'].loc[:105].sum()/phi_k3['phi(K)'].sum()