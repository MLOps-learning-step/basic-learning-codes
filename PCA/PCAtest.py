"""
PCA básico con sklearn — Wine Dataset
Versión corregida (bug fix: load_wine faltaba en el import original)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler

wine = load_wine()
X: np.ndarray = wine.data
y: np.ndarray = wine.target

scaler = StandardScaler()
X_std: np.ndarray = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca: np.ndarray = pca.fit_transform(X_std)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="viridis")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA — Wine Dataset")
plt.colorbar(label="Clase")
plt.show()
