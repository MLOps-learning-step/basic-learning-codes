#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  1 17:42:53 2026

@author: josedavidangaritapertuz
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '01_perceptron'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '02_adaline'))

from perceptron import Perceptron
from adaline_gd import AdalineGD
from adaline_sgd import AdalineSGD
from matplotlib.colors import ListedColormap


def plot_decision_regions(X, y, classifier, resolution=0.1):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    c=colors[idx],
                    marker=markers[idx], 
                    label=cl, 
                    edgecolor='black')


df = pd.read_csv('https://archive.ics.uci.edu/ml/'
        'machine-learning-databases/iris/iris.data', header=None)

#seleccionar setosa
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)

#
X = df.iloc[0:100, [0, 2]].values

# PERCEPTRON
ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)


# Adaline

ada1 = AdalineGD(n_iter= 10, eta= 0.01)

X_std = np.copy(X)
X_std[:,0] = (X[:,0] - X[:,0].mean())/ X[:,0].std()
X_std[:,1] = (X[:,1] - X[:,1].mean())/ X[:,1].std()

ada1.fit(X_std, y)
fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (10,4))
ax[0].plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), marker= 'o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Log(error cuadratico)')
ax[0].set_title('Adaline con n = 0.01')

ada2 = AdalineGD(n_iter= 10, eta= 0.000001).fit(X_std, y)
ax[1].plot(range(1, len(ada2.cost_) + 1), np.log10(ada2.cost_), marker= 'o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Log(error cuadratico)')
ax[1].set_title('Adaline con n = 0.00001')
plt.show()


ada3 = AdalineSGD(n_iter= 15, eta= 0.01, random_state=1)
ada3.fit(X_std, y)


plot_decision_regions(X_std, y, classifier=ada3)
plt.title('Adaline - Stochastic Gradient Descent')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='upper left')

plt.tight_layout()
# plt.savefig('images/02_15_1.png', dpi=300)
plt.show()

plt.plot(range(1, len(ada3.cost_) + 1), ada3.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Average Cost')

plt.tight_layout()
# plt.savefig('images/02_15_2.png', dpi=300)
plt.show()
