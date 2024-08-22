"""
Created on Aug 21, 2024
@author: Tixian Wang
"""

from importlib import resources

import matplotlib.pyplot as plt
import numpy as np
import numpy.random as npr

from assets import ASSETS, FILE_NAME
from neural_data_smoothing3D import coeff2posdir, coeff2strain, pos_dir_to_input

color = ["C" + str(i) for i in range(20)]


def main():

    with resources.path(ASSETS, FILE_NAME) as path:
        data = np.load(path, allow_pickle="TRUE").item()

    n_elem = data["model"]["n_elem"]
    L = data["model"]["L"]
    radius = data["model"]["radius"]
    s = data["model"]["s"]
    dl = data["model"]["dl"]
    nominal_shear = data["model"]["nominal_shear"]
    idx_data_pts = data["idx_data_pts"]
    # input_data = data['input_data']
    # true_pos = data['true_pos']
    # true_dir = data['true_dir']
    true_kappa = data["true_kappa"]
    pca = data["pca"]

    num_strain = len(pca)
    input_size = len(idx_data_pts) * (3 + 6)
    output_size = sum([pca[i].n_components for i in range(num_strain)])
    print("input_size:", input_size, "output_size:", output_size)
    coeffs = np.hstack(
        [pca[i].transform(true_kappa[:, i, :]) for i in range(num_strain)]
    )

    # for j in range(min(10, coeffs.shape[1])):
    # 	plt.figure(0)
    # 	plt.scatter([j]*len(coeffs), coeffs[:,j], color=color[j], s=20, marker='.')
    # # plt.show()
    # # quit()

    coeffs_mean = coeffs.mean(axis=0)
    coeffs_std = coeffs.std(axis=0)
    # coeffs_low = coeffs.min(axis=0)
    # coeffs_high = coeffs.max(axis=0)
    npr.seed(2024)
    n_training_data = int(1e4)
    coeffs_rand = (
        npr.randn(n_training_data, output_size) * coeffs_std + coeffs_mean
    )
    # coeffs_rand = npr.uniform(coeffs_low, coeffs_high, size=(n_training_data, output_size))

    # for j in range(min(10, coeffs.shape[1])):
    # 	plt.figure(1)
    # 	plt.scatter([j]*n_training_data, coeffs_rand[:,j], color=color[j], s=20, marker='.')
    # 	# plt.figure(2)
    # 	# plt.scatter(np.ones(n_training_data)*j, coeffs_rand2[:,j], color=color[j], s=20, marker='.')

    # # plt.show()

    strain_rand = coeff2strain(coeffs_rand, pca)
    # # print(strain_rand[0].shape, strain_rand[1].shape)
    # print(strain_rand.shape)
    posdir_rand = coeff2posdir(coeffs_rand, pca, dl, nominal_shear)
    # print(posdir_rand[0].shape, posdir_rand[1].shape)
    input_pos = posdir_rand[0][..., idx_data_pts]
    input_dir = posdir_rand[1][..., idx_data_pts]
    input_data = pos_dir_to_input(input_pos, input_dir)
    # output_dir = np.stack([input_data[:,3:6,:], np.cross(input_data[:,6:9,:], input_data[:,3:6,:], axis=1), input_data[:,6:9,:]], axis=2)
    # print(np.linalg.norm(input_dir - output_dir), input_dir[0,:,:,0], output_dir[0,:,:,0])
    # print(input_dir.shape, input_data.shape, output_dir.shape)
    # quit()

    idx_list = np.random.randint(
        n_training_data, size=10
    )  # [i*250 for i in range(10)]
    fig = plt.figure(2)
    ax = fig.add_subplot(111, projection="3d")
    fig2, axes = plt.subplots(ncols=3, sharex=True, figsize=(16, 5))
    for ii in range(len(idx_list)):
        i = idx_list[ii]
        ax.plot(
            posdir_rand[0][i, 0, :],
            posdir_rand[0][i, 1, :],
            posdir_rand[0][i, 2, :],
            ls="-",
            color=color[ii],
        )
        ax.scatter(
            input_data[i, 0, :],
            input_data[i, 1, :],
            input_data[i, 2, :],
            s=50,
            marker="o",
            color=color[ii],
        )
        ax.set_xlim(-L, 0)
        ax.set_ylim(-L, 0)
        ax.set_zlim(-L, 0)
        ax.set_aspect("equal")
        for j in range(3):
            axes[j].plot(s[1:-1], strain_rand[i, j, :])

    plt.show()

    flag_save = 1

    if flag_save:
        import os

        folder_name = "Data"
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)

        print("saving data...")
        training_data = {
            "coeffs": coeffs,
            "n_training_data": n_training_data,
            "input_size": input_size,
            "output_size": output_size,
            "coeffs_rand": coeffs_rand,
            "input_data": input_data,
            "true_pos": posdir_rand[0],
            "true_dir": posdir_rand[1],
            "true_kappa": strain_rand,
            "true_shear": nominal_shear,
        }
        np.save("Data/training_data_set.npy", training_data)


if __name__ == "__main__":
    main()
