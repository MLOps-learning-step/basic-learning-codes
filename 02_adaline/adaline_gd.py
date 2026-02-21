#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  6 11:52:11 2026

@author: josedavidangaritapertuz
"""

import numpy as np

class AdalineGD():
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta= eta
        self.n_iter = n_iter
        self.random_state = random_state
        
    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc= 0.0, scale= 0.01, size= 1 + X.shape[1])
        self.cost_ = []
        print(self.w_)
        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)        
            errors = (y-output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            print(cost)
            self.cost_.append(cost)
        return self
            
    def net_input(self,X):
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def activation(self, X):
        return X

    def predict(self,X):
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)
    
    def normalize(self, X):
        X_std = []
        
        X_std[:,0] = (X[:,0] - X[:,0].mean())/ X[:,0].std()
        X_std[:,1] = (X[:,1] - X[:,1].mean())/ X[:,1].std()
        
        return X_std