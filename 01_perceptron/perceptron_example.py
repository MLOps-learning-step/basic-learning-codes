#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 30 12:52:21 2025

@author: josedavidangaritapertuz
"""

import numpy as np

# Datos de ejemplo
X = np.array([[2.5, 3.0], [1.0, 1.5]])
y = np.array([1, -1])

# Pesos iniciales (simulados)
w_ = np.array([0.01, -0.02, 0.03])  # [bias, w1, w2]
eta = 0.1

print("=== Entrenamiento con zip() ===\n")

for i, (xi, target) in enumerate(zip(X, y)):
    print(f"Iteración {i}:")
    print(f"  xi (características) = {xi}")
    print(f"  target (etiqueta real) = {target}")
    
    # Predicción
    net_input = np.dot(xi, w_[1:]) + w_[0]
    prediction = 1 if net_input >= 0 else -1
    print(f"  net_input = {net_input:.4f}")
    print(f"  prediction = {prediction}")
    
    # Actualización
    update = eta * (target - prediction)
    print(f"  update = {eta} * ({target} - {prediction}) = {update}")
    
    if update != 0:
        w_[1:] += update * xi
        w_[0] += update
        print(f"  ¡Error! Pesos actualizados: {w_}")
    else:
        print(f"  Correcto, pesos sin cambio")
    print()
    
# Ejemplo simple
x = 10
assert x > 0  # Pasa sin problemas porque es True

# Con mensaje de error
precio = -50
assert precio >= 0, f"El precio no puede ser negativo: {precio}"
# AssertionError: El precio no puede ser negativo: -50