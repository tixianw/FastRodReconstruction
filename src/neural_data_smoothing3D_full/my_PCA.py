"""
Created on Aug 25, 2024
@author: Tixian Wang
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import eigh  # eig

# from tqdm import tqdm
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
color = ["C" + str(i) for i in range(10)]


class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.std = None

    def fit(self, X):
        """
        input X: original data, shape (samples, features)
        """
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        ## Center the data
        X = (X - self.mean) / (self.std + 1e-16)
        ## covariance matrix
        cov = np.cov(X.T)
        ## eigenvalues, eigenvectors of the covariance matrix
        eig_val, eig_vec = eigh(cov)  # eig(cov)
        # print(eig_val.dtype, eig_vec.dtype)
        # # cast to real values if the imaginary parts are in floating precision to 0
        # eig_val = np.real_if_close(eig_val)
        # eig_vec = np.real_if_close(eig_vec)
        # print(eig_val.dtype, eig_vec.dtype)
        # # round values that are at the floating point precision to exactly 0
        # small = np.abs(eig_val) < 2 * np.finfo(eig_val.dtype).eps
        # eig_val[small] = 0

        ## Sort eigenvalues
        idx = np.argsort(eig_val)[::-1]
        eig_val = eig_val[idx]
        eig_vec = eig_vec[:, idx]
        print(eig_val[: self.n_components].sum() / eig_val.sum())
        ## feature reductino
        self.components = eig_vec[:, : self.n_components]

    def transform(self, X):
        X = (X - self.mean) / self.std
        X_projected = X @ self.components
        return X_projected

    def reduce(self, X):
        X_hat = self.transform(X) @ self.components.T * self.std + self.mean
        return X_hat

    def approximate(self, coeff):
        X_hat = coeff @ self.components.T * self.std + self.mean
        return X_hat

