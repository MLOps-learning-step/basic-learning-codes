#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  6 14:49:46 2026

@author: josedavidangaritapertuz
"""

import numpy as np

class AdalineSGD():
    def __init__(self, eta = 0.01, n_iter = 10, shuffle = True, random_state = None):
        self.eta = eta
        self.n_iter = n_iter
        self.shuffle = shuffle
        self.random_state = random_state

    def fit(self, X, y):
        self.initialize_weights(X.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            cost = []
            for xi, target in zip(X, y):
                cost.append(self._update_weight(xi, target))
            avg_cost = sum(cost) / len(y)
            self.cost_.append(avg_cost)
        return self
    
    def partial_fit(self, X, y):
        if not self.w_initialized:
            self.initialize_weights(x.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X,y):
                self._update_weight(xi, target)
        else:
            self._update_weight(X, y)
        return selfßå
            
            
    def _update_weight(self, xi, target):
        output = self.activation(self.net_input(xi))
        error = (target - output)
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        cost = 0.5 * error**2
        return cost

    def net_input(self, X):
        """ Es lo que entra al principio de la neurona"""
        return np.dot(X, self.w_[1:])+self.w_[0]
    
    def activation(self, X):
        """Va a ser lo que me permite recalcular los pesos, esta ves es 1"""
        return X
    
    def predict(self, X):
        """Es la salida final de mi neurona"""
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)
                
    def _shuffle(self, X, y):
        r = self.rgen.permutation(len(y))
        return X[r], y[r]
    
        
    def initialize_weights(self, m):
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc=0.0, scale= 0.01, size= 1 + m)
        self.w_initialized = True