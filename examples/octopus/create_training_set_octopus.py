"""
Created on Aug 21, 2024
@author: Tixian Wang
"""

from importlib import resources

import matplotlib.pyplot as plt
import numpy as np
import numpy.random as npr

from assets import ASSETS
from assets import FILE_NAME_OCTOPUS as FILE_NAME
from neural_data_smoothing3D_full import coeff2posdir, coeff2strain, pos_dir_to_input
from neural_data_smoothing3D_full.utils import _aver

color = ["C" + str(i) for i in range(20)]


def main():

	with resources.path(ASSETS, FILE_NAME) as path:
		print('Reading file', FILE_NAME, '...')
		data = np.load(path, allow_pickle="TRUE").item()
	
	print('number of markers (excluding the base):', data['n_data_pts'])

	n_elem = data["model"]["n_elem"]
	L = data["model"]["L"]
	radius = data["model"]["radius"]
	s = data["model"]["s"]
	s_mean = _aver(s)
	dl = data["model"]["dl"]
	nominal_shear = data["model"]["nominal_shear"]
	idx_data_pts = data["idx_data_pts"]
	# input_data = data['input_data']
	# true_pos = data['true_pos']
	# true_dir = data['true_dir']
	true_kappa = data["true_kappa"]
	true_shear = data['true_shear']
	pca = data["pca"]

	num_strain = len(pca)
	input_size = len(idx_data_pts) * (3 + 6)
	output_size = sum([pca[i].n_components for i in range(num_strain)])
	print("input_size:", input_size, "output_size:", output_size)
	coeffs = np.hstack(
		[pca[i].transform(true_kappa[:, i, :]) for i in range(3)] + \
		[pca[i+3].transform(true_shear[:, i, :]) for i in range(3)]
	)

	# n_check = 9
	# for j in range(n_check):
	# 	plt.figure(0)
	# 	plt.scatter([j]*len(coeffs), coeffs[:,j], color=color[j], s=20, marker='.')
	# 	plt.figure(1)
	# 	plt.scatter([j+n_check]*len(coeffs), coeffs[:,j+n_check], color=color[j], s=20, marker='.')
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

	# for j in range(n_check):
	# 	plt.figure(2)
	# 	plt.scatter([j]*n_training_data, coeffs_rand[:,j], color=color[j], s=20, marker='.')
	# 	plt.figure(3)
	# 	plt.scatter([j+n_check]*n_training_data, coeffs_rand[:,j+n_check], color=color[j], s=20, marker='.')

	# plt.show()
	# quit()

	strain_rand = coeff2strain(coeffs_rand, pca)
	# # print(strain_rand[0].shape, strain_rand[1].shape)
	posdir_rand = coeff2posdir(coeffs_rand, pca, dl)
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
	fig2, axes = plt.subplots(ncols=3, nrows=2, sharex=True, figsize=(16, 5))
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
			axes[0][j].plot(s[1:-1], strain_rand[0][i, j, :])
			axes[1][j].plot(s_mean, strain_rand[1][i, j, :])
			axes[0][j].set_ylabel('$\\kappa_%d$'%(j+1))
			axes[1][j].set_ylabel('$\\nu_%d$'%(j+1))

	flag_save = 0

	if flag_save:
		import os

		folder_name = "assets"
		if not os.path.exists(folder_name):
			os.mkdir(folder_name)

		print("saving data to file", FILE_NAME, '...')

		training_data = {
			"coeffs": coeffs,
			"n_training_data": n_training_data,
			"input_size": input_size,
			"output_size": output_size,
			"coeffs_rand": coeffs_rand,
			"input_data": input_data,
			"true_pos": posdir_rand[0],
			"true_dir": posdir_rand[1],
			"true_kappa": strain_rand[0],
			"true_shear": strain_rand[1],
		}
		np.save(folder_name + "/training_data_set_octopus.npy", training_data)
	
	plt.show()


if __name__ == "__main__":
	main()
