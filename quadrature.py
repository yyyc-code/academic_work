# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 11:58:44 2019

@author: mdejg
"""

import numpy as np
from itertools import product

class quadrature:
    def __init__(self,func,method,*,a ,b ,N,epsilon = 0,arg = 'Laguerre',multivar = []):
        self.a = a
        self.b = b
        self.N = N
        self.epsilon = epsilon
        self.func = func
        self.arg = arg
        self.multivar = multivar
        if multivar == []:
            if method == 'Gauss':
                self.gauss_method()
            elif method == 'Riemann':
                self.riemann_method()
            else:
                raise TypeError('Undefined Method!')     

        else:
            self.get_nodes_multivar()        
        self.integral = self.get_integral() 
            
    def get_integral(self):
        self.get_fx()
        recta = self.fx*self.weights
        return recta.sum()
    
    def get_fx(self):
        self.fx = self.func(self.nodes)
        
    
    def riemann_method(self):
        i = np.arange(0,self.N)
        self.weights = (self.b-self.a)/self.N
        self.nodes = self.a + (i+self.epsilon)*self.weights
        
        
    def gauss_method(self):
        if self.arg == 'Hermite':
            inte = np.polynomial.hermite.hermgauss(self.N)
        elif self.arg == 'Legendre':
            inte = np.polynomial.legendre.leggauss(self.N)
        elif self.arg == 'Laguerre':
            inte = np.polynomial.laguerre.laggauss(self.N)
        self.nodes = inte[0]*(self.b-self.a)/2+(self.b+self.a)/2
        self.weights = inte[1]*(self.b-self.a)/2
    
    def get_nodes_multivar(self):
        i = np.arange(0,self.N)
        self.weights = ((self.b-self.a)/self.N)*((self.multivar[1]-self.multivar[0])/self.N)
        x = self.a + (i+self.epsilon)*((self.b-self.a)/self.N)
        y = self.multivar[0]+ (i+self.epsilon)*((self.multivar[1]-self.multivar[0])/self.N)
        self.nodes = list(product(x,y))

        
        
        

        