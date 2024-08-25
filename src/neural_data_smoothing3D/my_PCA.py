"""
Created on Aug 5, 2024
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


def main():
    folder_name = "Data/"
    file_name = "BR2_arm_data"
    data = np.load(folder_name + file_name + ".npy", allow_pickle="TRUE").item()

    n_elem = data["model"]["n_elem"]
    L = data["model"]["L"]
    radius = data["model"]["radius"]
    s = data["model"]["s"]
    dl = data["model"]["dl"]
    nominal_shear = data["model"]["nominal_shear"]
    # idx_data_pts = data['idx_data_pts']
    # input_data = data['input_data']
    # true_pos = data['true_pos']
    # true_dir = data['true_dir']
    true_kappa = data["true_kappa"]

    n_components = np.array([3, 3, 3])
    pca_list = []
    for i in range(len(n_components)):
        pca = PCA(n_components=n_components[i])
        pca.fit(true_kappa[:, i, :])
        pca_list.append(pca)

    flag_save = 0

    if flag_save:
        print("saving data...")

        data["pca"] = pca_list
        np.save("Data/" + file_name + ".npy", data)

    # np.random.seed(2024)
    # idx_list = np.random.randint(len(true_kappa), size=10) # [i*10 for i in range(10)]
    # pos_approximate = curvature2position(kappa_approximate)
    # for ii in range(len(idx_list)):
    # 	i = idx_list[ii]
    # 	plt.figure(1)
    # 	plt.plot(s[1:-1], true_kappa[i,:], color=color[ii], ls='-')
    # 	plt.plot(s[1:-1], kappa_approximate[i,:], color=color[ii], ls='--')
    # 	plt.figure(2)
    # 	plt.plot(pos_approximate[i,0,:], pos_approximate[i,1,:], color=color[ii], ls='-')
    # 	plt.plot(true_pos[i,0,:], true_pos[i,1,:], color=color[ii], ls='--')
    # 	plt.scatter(input_data[i,0,:], input_data[i,1,:], s=50, marker='o', color=color[ii])

    for i in range(len(n_components)):
        print(i, pca_list[i].mean.mean(), pca_list[i].std.std())
        plt.figure(i)
        for j in range(n_components[i]):
            plt.plot(s[1:-1], pca_list[i].components[:, j])

    plt.show()


if __name__ == "__main__":
    main()
    # main(sys.argv[1])
