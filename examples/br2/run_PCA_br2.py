"""
Created on Aug 25, 2024
@author: Tixian Wang
"""

from importlib import resources

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import eigh  # eig

from assets import ASSETS, FILE_NAME_BR2
FILE_NAME = FILE_NAME_BR2[4]
from neural_data_smoothing3D import PCA, pos_dir_to_input

color = ["C" + str(i) for i in range(10)]


def main():
    with resources.path(ASSETS, FILE_NAME) as path:
        data = np.load(path, allow_pickle="TRUE").item()

    L = data["model"]["L"]
    print('rest length:', L)
    s = data["model"]["s"]
    
    ## data point setup
    n_data_pts = 3 # exlude the initial point at base
    idx_data_pts = np.array(
        [int(100 / (n_data_pts)) * i for i in range(1, n_data_pts)] + [-1]
    )
    print("idx of s_j's:", idx_data_pts)

    position = data["true_pos"]
    director = data["true_dir"]
    input_pos = position[..., idx_data_pts]
    input_dir = director[..., idx_data_pts]
    input_data = pos_dir_to_input(input_pos, input_dir)

    true_kappa = data["true_kappa"]

    n_components = np.array([3, 3, 3])
    pca_list = []
    for i in range(len(n_components)):
        pca = PCA(n_components=n_components[i])
        pca.fit(true_kappa[:, i, :])
        pca_list.append(pca)

    flag_save = 0

    if flag_save:
        import os

        folder_name = "assets"
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)

        print("saving data...")

        data["n_data_pts"] = n_data_pts
        data["idx_data_pts"] = idx_data_pts
        data["input_data"] = input_data
        data["pca"] = pca_list
        np.save(folder_name + '/' + FILE_NAME, data)

    for i in range(len(n_components)):
        plt.figure(i)
        for j in range(n_components[i]):
            plt.plot(s[1:-1], pca_list[i].components[:, j])

    plt.show()


if __name__ == "__main__":
    main()
