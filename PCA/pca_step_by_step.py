"""
PCA Paso a Paso - Basado en el libro "Python Machine Learning" de Sebastian Raschka
================================================================================
Capítulo 5: Compressing Data via Dimensionality Reduction

El flujo completo de PCA (implementación manual) es:
    1. Estandarizar los datos (media=0, varianza=1)
    2. Construir la matriz de covarianza
    3. Descomponer la matriz de covarianza en eigenvectores y eigenvalores
    4. Ordenar eigenvalores de mayor a menor y seleccionar los k eigenvectores
    5. Construir la matriz de proyección W a partir de los k eigenvectores
    6. Transformar el dataset original X usando W → X' = X · W

Dataset: Wine (el mismo que usa Raschka en el libro)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as SklearnPCA

# =============================================================================
# CARGA DE DATOS
# =============================================================================
wine = load_wine()
X: np.ndarray = wine.data          # shape: (178, 13) — 178 muestras, 13 features
y: np.ndarray = wine.target        # 3 clases: 0, 1, 2
feature_names: list[str] = wine.feature_names
class_names: list[str] = list(wine.target_names)

print("=" * 65)
print("DATASET WINE")
print("=" * 65)
print(f"  Muestras   : {X.shape[0]}")
print(f"  Features   : {X.shape[1]}")
print(f"  Clases     : {class_names}")
print()

# =============================================================================
# PASO 1: ESTANDARIZACIÓN
# =============================================================================
# Por qué: PCA es sensible a la escala de las variables. Si no estandarizamos,
# las variables con mayor rango dominarán los componentes principales.
# Transformamos cada feature para que tenga media=0 y desviación estándar=1.
# =============================================================================
print("=" * 65)
print("PASO 1: Estandarización de los datos")
print("=" * 65)

scaler = StandardScaler()
X_std: np.ndarray = scaler.fit_transform(X)

print(f"  Media antes  (feature 0): {X[:, 0].mean():.4f}")
print(f"  Media después (feature 0): {X_std[:, 0].mean():.4f}  ← ~0")
print(f"  Std antes  (feature 0): {X[:, 0].std():.4f}")
print(f"  Std después (feature 0): {X_std[:, 0].std():.4f}  ← ~1")
print()

# =============================================================================
# PASO 2: MATRIZ DE COVARIANZA
# =============================================================================
# Por qué: La matriz de covarianza captura las relaciones (correlaciones lineales)
# entre todas las pares de features. Es una matriz simétrica de forma (d x d)
# donde d es el número de features (13 en este caso).
#
# Fórmula: Σ = (1 / n-1) · Xᵀ · X  (usando datos centrados)
# =============================================================================
print("=" * 65)
print("PASO 2: Construcción de la Matriz de Covarianza (13x13)")
print("=" * 65)

cov_matrix: np.ndarray = np.cov(X_std.T)   # np.cov espera (features, muestras)

print(f"  Forma de la matriz de covarianza: {cov_matrix.shape}")
print(f"  Diagonal (varianza de cada feature, debe ser ~1.0):")
variances = np.diag(cov_matrix)
print(f"    {variances.round(4)}")
print()

# =============================================================================
# PASO 3: EIGENDESCOMPOSICIÓN
# =============================================================================
# Por qué: Los eigenvectores de la matriz de covarianza definen las direcciones
# (ejes) a lo largo de las cuales los datos tienen mayor varianza.
# Los eigenvalores asociados indican cuánta varianza captura cada dirección.
#
# Propiedad clave: Σ · v = λ · v
#   donde v = eigenvector, λ = eigenvalor
# =============================================================================
print("=" * 65)
print("PASO 3: Eigendescomposición de la Matriz de Covarianza")
print("=" * 65)

eigenvalues: np.ndarray
eigenvectors: np.ndarray
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Nota: eigenvectors[:, i] es el i-ésimo eigenvector (columnas, no filas)
print(f"  Número de eigenvalores: {len(eigenvalues)}")
print(f"  Eigenvalores (sin ordenar):")
for i, ev in enumerate(eigenvalues):
    print(f"    λ{i+1:02d} = {ev:.4f}")
print()

# Verificación: suma de eigenvalores == suma de varianzas (traza de Σ)
print(f"  Suma eigenvalores  : {eigenvalues.sum():.4f}")
print(f"  Traza(Σ) (∑variances): {np.trace(cov_matrix):.4f}  ← deben coincidir")
print()

# =============================================================================
# PASO 4: ORDENAR EIGENVALORES Y CALCULAR VARIANZA EXPLICADA
# =============================================================================
# Por qué: Seleccionamos los k eigenvectores con MAYOR eigenvalor porque
# son las direcciones que concentran más información (varianza) del dataset.
# Descartar las direcciones con menor varianza = reducir dimensionalidad.
# =============================================================================
print("=" * 65)
print("PASO 4: Ordenar por eigenvalor descendente y varianza explicada")
print("=" * 65)

# Indices ordenados de mayor a menor eigenvalor
sorted_idx: np.ndarray = np.argsort(eigenvalues)[::-1]
eigenvalues_sorted: np.ndarray = eigenvalues[sorted_idx]
eigenvectors_sorted: np.ndarray = eigenvectors[:, sorted_idx]

# Varianza explicada por cada componente principal
total_variance: float = eigenvalues_sorted.sum()
explained_variance_ratio: np.ndarray = eigenvalues_sorted / total_variance
cumulative_variance: np.ndarray = np.cumsum(explained_variance_ratio)

print(f"  {'PC':<5} {'Eigenvalor':>12} {'Var. Explicada':>15} {'Var. Acumulada':>15}")
print(f"  {'-'*5} {'-'*12} {'-'*15} {'-'*15}")
for i in range(len(eigenvalues_sorted)):
    print(
        f"  PC{i+1:<3} {eigenvalues_sorted[i]:>12.4f} "
        f"{explained_variance_ratio[i]:>14.2%} "
        f"{cumulative_variance[i]:>14.2%}"
    )
print()
print(f"  Los 2 primeros PCs explican: {cumulative_variance[1]:.2%} de la varianza total")
print()

# =============================================================================
# PASO 5: CONSTRUCCIÓN DE LA MATRIZ DE PROYECCIÓN W
# =============================================================================
# Por qué: Tomamos los k eigenvectores con mayor eigenvalor y los apilamos
# como columnas para formar la matriz W de forma (d x k).
# W define el nuevo subespacio de dimensión k al que proyectaremos los datos.
# =============================================================================
print("=" * 65)
print("PASO 5: Construcción de la Matriz de Proyección W (13x2)")
print("=" * 65)

k: int = 2  # Reducimos a 2 dimensiones
W: np.ndarray = eigenvectors_sorted[:, :k]

print(f"  Forma de W: {W.shape}  (13 features → {k} componentes principales)")
print(f"  PC1 (eigenvector con λ={eigenvalues_sorted[0]:.4f}):")
print(f"    {W[:, 0].round(4)}")
print(f"  PC2 (eigenvector con λ={eigenvalues_sorted[1]:.4f}):")
print(f"    {W[:, 1].round(4)}")
print()

# =============================================================================
# PASO 6: TRANSFORMACIÓN — PROYECCIÓN DE LOS DATOS
# =============================================================================
# Por qué: Multiplicamos los datos estandarizados por W para obtener las
# nuevas coordenadas en el subespacio de los componentes principales.
#
# Fórmula: X' = X_std · W    shape: (178, 13) · (13, 2) = (178, 2)
# =============================================================================
print("=" * 65)
print("PASO 6: Transformación X' = X_std · W")
print("=" * 65)

X_pca_manual: np.ndarray = X_std.dot(W)

print(f"  Forma original (estandarizada): {X_std.shape}")
print(f"  Forma reducida (PCA manual)   : {X_pca_manual.shape}")
print()

# =============================================================================
# VALIDACIÓN: Comparar resultado manual vs sklearn
# =============================================================================
print("=" * 65)
print("VALIDACIÓN: PCA Manual vs sklearn PCA")
print("=" * 65)

sklearn_pca = SklearnPCA(n_components=2)
X_pca_sklearn: np.ndarray = sklearn_pca.fit_transform(X_std)

# Los signos de los eigenvectores pueden estar invertidos (ambos son válidos)
# Comparamos la correlación absoluta entre las proyecciones
corr_pc1: float = abs(np.corrcoef(X_pca_manual[:, 0], X_pca_sklearn[:, 0])[0, 1])
corr_pc2: float = abs(np.corrcoef(X_pca_manual[:, 1], X_pca_sklearn[:, 1])[0, 1])

print(f"  Correlación PC1 (manual vs sklearn): {corr_pc1:.6f}  ← debe ser ~1.0")
print(f"  Correlación PC2 (manual vs sklearn): {corr_pc2:.6f}  ← debe ser ~1.0")
print()
print(f"  Varianza explicada sklearn:")
for i, ratio in enumerate(sklearn_pca.explained_variance_ratio_):
    print(f"    PC{i+1}: {ratio:.4f}  vs manual: {explained_variance_ratio[i]:.4f}")
print()

# =============================================================================
# VISUALIZACIONES
# =============================================================================

colors: list[str] = ["#1f77b4", "#ff7f0e", "#2ca02c"]
markers: list[str] = ["o", "s", "^"]

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle(
    "PCA Paso a Paso — Wine Dataset (Raschka, Python Machine Learning)",
    fontsize=14, fontweight="bold", y=1.01
)

# ── Gráfico 1: Varianza Explicada (Scree Plot) ────────────────────────────────
ax1 = axes[0, 0]
pcs = [f"PC{i+1}" for i in range(len(eigenvalues_sorted))]
ax1.bar(pcs, explained_variance_ratio, alpha=0.7, color="#1f77b4", label="Individual")
ax1.step(pcs, cumulative_variance, where="mid", color="#ff7f0e",
         linewidth=2, label="Acumulada")
ax1.axhline(y=0.80, color="red", linestyle="--", alpha=0.6, label="80% umbral")
ax1.set_xlabel("Componente Principal")
ax1.set_ylabel("Ratio de Varianza Explicada")
ax1.set_title("Paso 4: Scree Plot — Varianza Explicada")
ax1.legend(loc="center right")
ax1.set_xticks(range(len(pcs)))
ax1.set_xticklabels(pcs, rotation=45, ha="right", fontsize=7)
ax1.set_ylim(0, 1.05)

# ── Gráfico 2: Proyección PCA Manual ─────────────────────────────────────────
ax2 = axes[0, 1]
for label, color, marker in zip(range(3), colors, markers):
    mask = y == label
    ax2.scatter(
        X_pca_manual[mask, 0], X_pca_manual[mask, 1],
        c=color, marker=marker, label=class_names[label],
        alpha=0.8, edgecolors="white", linewidths=0.5
    )
ax2.set_xlabel(f"PC1 ({explained_variance_ratio[0]:.1%} varianza)")
ax2.set_ylabel(f"PC2 ({explained_variance_ratio[1]:.1%} varianza)")
ax2.set_title("Paso 6: Proyección PCA (Implementación Manual)")
ax2.legend()
ax2.grid(True, alpha=0.3)

# ── Gráfico 3: Proyección PCA sklearn ────────────────────────────────────────
ax3 = axes[1, 0]
for label, color, marker in zip(range(3), colors, markers):
    mask = y == label
    ax3.scatter(
        X_pca_sklearn[mask, 0], X_pca_sklearn[mask, 1],
        c=color, marker=marker, label=class_names[label],
        alpha=0.8, edgecolors="white", linewidths=0.5
    )
ax3.set_xlabel(f"PC1 ({sklearn_pca.explained_variance_ratio_[0]:.1%} varianza)")
ax3.set_ylabel(f"PC2 ({sklearn_pca.explained_variance_ratio_[1]:.1%} varianza)")
ax3.set_title("Validación: Proyección PCA (sklearn)")
ax3.legend()
ax3.grid(True, alpha=0.3)

# ── Gráfico 4: Biplot — Eigenvectores sobre datos proyectados ────────────────
# Muestra la contribución de cada feature original a los componentes principales
ax4 = axes[1, 1]
for label, color, marker in zip(range(3), colors, markers):
    mask = y == label
    ax4.scatter(
        X_pca_sklearn[mask, 0], X_pca_sklearn[mask, 1],
        c=color, marker=marker, label=class_names[label],
        alpha=0.4, edgecolors="white", linewidths=0.5, s=30
    )

# Escalar los loadings para que sean visibles en el mismo espacio
scale: float = np.max(np.abs(X_pca_sklearn)) * 0.5
loadings: np.ndarray = sklearn_pca.components_.T  # shape (13, 2)

for i, feature in enumerate(feature_names):
    ax4.annotate(
        "", xy=(loadings[i, 0] * scale, loadings[i, 1] * scale),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle="->", color="red", lw=1.5)
    )
    ax4.text(
        loadings[i, 0] * scale * 1.12,
        loadings[i, 1] * scale * 1.12,
        feature, fontsize=6.5, color="darkred", ha="center"
    )

ax4.set_xlabel("PC1")
ax4.set_ylabel("PC2")
ax4.set_title("Biplot: Contribución de features a los PCs")
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)
ax4.axhline(0, color="gray", linewidth=0.5)
ax4.axvline(0, color="gray", linewidth=0.5)

plt.tight_layout()
plt.savefig("PCA/pca_wine_analysis.png", dpi=150, bbox_inches="tight")
plt.show()

print("=" * 65)
print("RESUMEN DEL FLUJO PCA (Raschka, Cap. 5)")
print("=" * 65)
print("  1. Estandarizar X → media=0, std=1")
print("  2. Calcular Σ = (1/n-1) · Xᵀ · X  (covarianza)")
print("  3. Eigendescomposición: Σ·v = λ·v")
print("  4. Ordenar λ desc → seleccionar top-k eigenvectores")
print("  5. Construir W = [v₁ | v₂ | ... | vₖ]  (13×2)")
print("  6. Proyectar: X' = X_std · W  (178×13 · 13×2 = 178×2)")
print()
print("  Con k=2: se conserva el {:.1f}% de la varianza total".format(
    cumulative_variance[1] * 100
))
print("=" * 65)
