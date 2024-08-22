"""
Created on Jun 18, 2024
@author: Tixian Wang
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import eig
from tqdm import tqdm

# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
from utils import _aver_kernel, coeff2curvature, curvature2position

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
        X = (X - self.mean) / self.std
        ## covariance matrix
        cov = np.cov(X.T)
        ## eigenvalues, eigenvectors of the covariance matrix
        eig_val, eig_vec = eig(cov)
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


def main(choice):
    if choice == "0":
        folder_name = "Data/fake_br2_arm_data/"

        ## simulation data
        n_cases = 12
        # time_skip = 0.1
        step_skip = 1

        ## data point setup
        n_data_pts = 8
        idx_data_pts = np.array(
            [int(100 / (n_data_pts - 1)) * i for i in range(1, n_data_pts - 1)]
            + [-1]
        )

        # print('idx of data pts', idx_data_pts)

        input_data = []
        output_data = []
        true_pos = []
        true_kappa = []

        # plt.figure(1)
        for i in range(n_cases):
            file_name = "arm_data" + str(i)
            data = np.load(
                folder_name + file_name + ".npy", allow_pickle="TRUE"
            ).item()
            if i == 0:
                n_elem = data["model"]["arm"]["n_elem"]
                L = data["model"]["arm"]["L"]
                radius = data["model"]["arm"]["radius"]
                E = data["model"]["arm"]["E"]
                dt = data["model"]["numerics"]["step_size"]
                s = data["model"]["arm"]["s"]
                s_mean = (s[1:] + s[:-1]) / 2
                ds = s[1] - s[0]
            t = data["t"]
            final_time = t[-1]  # data['model']['numerics']['final_time']
            arm = data["arm"]
            position = arm[-1]["position"][:, :, :]
            orientation = arm[-1]["orientation"]
            kappa = arm[-1]["kappa"][:, :]
            # kappa_derivative = _aver_kernel(np.diff(kappa)/ds)
            # output = np.hstack([kappa[..., [0]], kappa_derivative])
            output = kappa.copy()
            # print(position[::step_skip,:-1,idx_data_pts].shape)
            input_data.append(position[::step_skip, :-1, idx_data_pts])
            output_data.append(output[::step_skip, :])
            true_pos.append(position[::step_skip, :-1, :])
            true_kappa.append(kappa[::step_skip, :])

        input_data = np.vstack(input_data)  # .reshape(-1, 2*n_data_pts)
        output_data = np.vstack(output_data)
        true_pos = np.vstack(true_pos)  # .reshape(-1,2*(n_elem+1))
        true_kappa = np.vstack(true_kappa)
        # print(input_data.shape, output_data.shape, true_pos.shape, true_kappa.shape)

        n_components = 3  # 14 # 10 # 5
        pca = PCA(n_components=n_components)
        pca.fit(output_data)  # [..., 1:]
        coeffs = pca.transform(
            output_data
        )  # np.hstack([output_data[..., [0]], pca.transform(output_data[..., 1:])])
        kappa_approximate = coeff2curvature(coeffs, pca)

        flag_save = 0

        if flag_save:
            print("saving data...")

            model_data = {
                "n_elem": n_elem,
                "L": L,
                "radius": radius,
                "E": E,
                "dt": dt,
                "s": s,
            }

            data = {
                "model": model_data,
                # 'n_data_pts': n_data_pts,
                "idx_data_pts": idx_data_pts,
                "input_data": input_data,
                "output_data": output_data,
                "true_pos": true_pos,
                "true_kappa": true_kappa,
                "pca": pca,
            }

            np.save("Data/fake_br2_arm_data1.npy", data)

    elif choice == "1":
        folder_name = "Data/"
        file_name = "pyelastica_arm_data"
        data = np.load(
            folder_name + file_name + ".npy", allow_pickle="TRUE"
        ).item()

        n_elem = data["model"]["n_elem"]
        L = data["model"]["L"]
        radius = data["model"]["radius"]
        s = data["model"]["s"]
        ds = s[1] - s[0]
        idx_data_pts = data["idx_data_pts"]
        input_data = data["input_data"]
        true_pos = data["true_pos"]
        true_kappa = data["true_kappa"]

        kappa_derivative = _aver_kernel(np.diff(true_kappa) / ds)
        output_data = np.hstack([true_kappa[..., [0]], kappa_derivative])

        n_components = 13  # 14 # 10 # 5
        pca = PCA(n_components=n_components)
        pca.fit(output_data[..., 1:])
        coeffs = np.hstack(
            [output_data[..., [0]], pca.transform(output_data[..., 1:])]
        )
        kappa_approximate = coeff2curvature(coeffs, pca)

        flag_save = 0

        if flag_save:
            print("saving data...")

            data["output_data"] = output_data
            data["pca"] = pca
            np.save("Data/" + file_name + "2.npy", data)

    np.random.seed(2024)
    idx_list = np.random.randint(
        len(true_kappa), size=10
    )  # [i*10 for i in range(10)]
    pos_approximate = curvature2position(kappa_approximate)
    for ii in range(len(idx_list)):
        i = idx_list[ii]
        plt.figure(1)
        plt.plot(s[1:-1], true_kappa[i, :], color=color[ii], ls="-")
        plt.plot(s[1:-1], kappa_approximate[i, :], color=color[ii], ls="--")
        plt.figure(2)
        plt.plot(
            pos_approximate[i, 0, :],
            pos_approximate[i, 1, :],
            color=color[ii],
            ls="-",
        )
        plt.plot(true_pos[i, 0, :], true_pos[i, 1, :], color=color[ii], ls="--")
        plt.scatter(
            input_data[i, 0, :],
            input_data[i, 1, :],
            s=50,
            marker="o",
            color=color[ii],
        )

    plt.figure(0)
    for i in range(n_components):
        plt.plot(s[1:-1], pca.components[:, i])

    plt.show()


if __name__ == "__main__":
    # main()
    main(sys.argv[1])
