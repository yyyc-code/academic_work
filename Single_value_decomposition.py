# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 20:40:03 2019

@author: mdejg
"""

import pandas as pd
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

#parse data

ticker = pd.read_excel('sp100.xlsx').ticker

price = pd.DataFrame(dtype=float)

for tic in ticker:
    p = pd.read_csv('%s.csv' %tic,index_col=0)['Adj Close']
    p.columns = ['%s' %tic]
    price = price.join(p.rename('%s' %tic),how = 'outer')
    
# five year data

price = price.loc['2013-01-01':]

#only two stocks have missive missing values, other stocks have no missing values
count = price.isnull().sum(axis=0)
dropp = count[count!=0].index.values

price = price.drop(dropp,axis=1) #clean data

rets = np.log(price/price.shift(1)).dropna()

C = rets.cov()

eig_vals, eig_vecs  = np.linalg.eig(C)  
eig_vals_sorted = np.sort(eig_vals)[::-1]  
eig_vecs_sorted = eig_vecs[:, eig_vals.argsort()[::-1]] 

scipy.stats.describe(eig_vals)
b = (eig_vals_sorted/eig_vals_sorted.sum()).cumsum()

pd.Series(b).plot(figsize = (8,6))
plt.xlabel('Count')
plt.ylabel('Percentage')

u,d,v = np.linalg.svd(C)
C_inv = u@(np.diag(1/d))@v
G = np.matrix([np.ones(100),[1]*17+[0]*83])
c = np.matrix([[1],[0.1]])
R = np.matrix(rets.mean()).T
A = G@C_inv@G.T
lamb = np.linalg.inv(A)@(G@C_inv@R-2*c)

w = 0.5*C_inv@(R-G.T@lamb)
