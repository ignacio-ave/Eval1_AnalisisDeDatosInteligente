
# My Utility : auxiliars functions

import pandas as pd
import numpy  as np

# ---------------------------------------------------------
# CResidual-Dispersion Entropy
# ---------------------------------------------------------
def entropy_dispersion(x, d, tau, c):
    """
    Calcula la Entropía de Dispersión (DE).
    Parámetros:
    x   : array-like, serie temporal
    d   : dimensión de embedding
    tau : retardo
    c   : número de símbolos
    """
    x = np.array(x, dtype=float)
    N = len(x)

    # Paso 1: Normalizar entre [0,1]
    x_norm = (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-12)

    # Paso 2: Embedding
    M = N - (d - 1) * tau
    if M <= 0:
        raise ValueError("Serie demasiado corta para embedding")
    X_emb = np.array([x_norm[i:i + d * tau:tau] for i in range(M)])

    # Paso 3: Simbolización
    Y = np.round(0.5 + X_emb * c).astype(int)
    Y[Y < 1] = 1
    Y[Y > c] = c

    # Paso 4: Patrones (convertir cada vector en número base-c)
    patrones = []
    for y in Y:
        code = 0
        for j in range(d):
            code += (y[j] - 1) * (c ** j)
        patrones.append(code)
    patrones = np.array(patrones)

    # Paso 5: Frecuencias
    unique, counts = np.unique(patrones, return_counts=True)
    probs = counts / M

    # Paso 6: Entropía de Shannon
    entr = -np.sum(probs * np.log(probs + 1e-12))

    # Paso 7: Normalización
    entr = entr / np.log(c ** d)

    return entr


# ---------------------------------------------------------
# Permutation Entropy
# ---------------------------------------------------------
def entropy_permuta(x, m, tau):
    """
    Calcula la Entropía de Permutación (PE).
    Parámetros:
    x   : array-like, serie temporal
    m   : dimensión de embedding
    tau : retardo
    """
    x = np.array(x, dtype=float)
    N = len(x)

    # Paso 1: Embedding
    M = N - (m - 1) * tau
    if M <= 0:
        raise ValueError("Serie demasiado corta para embedding")
    X_emb = np.array([x[i:i + m * tau:tau] for i in range(M)])

    # Paso 2: Patrones ordinales
    patrones = []
    for row in X_emb:
        perm = tuple(np.argsort(row))  # orden relativo
        patrones.append(perm)
    patrones = np.array(patrones)

    # Paso 3: Frecuencias
    unique, counts = np.unique(patrones, axis=0, return_counts=True)
    probs = counts / M

    # Paso 4: Entropía de Shannon
    entr = -np.sum(probs * np.log(probs + 1e-12))

    # Paso 5: Normalización
    from math import factorial, log
    entr = entr / log(factorial(m))

    return entr
