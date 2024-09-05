"""
Created on Sep 4, 2024
@author: Tixian Wang
"""

from importlib import resources

import matplotlib.pyplot as plt
import numpy as np

from assets import ASSETS
from assets import FILE_NAME_OCTOPUS_H5 as FILE_NAME
from neural_data_smoothing3D_full import PCA, pos_dir_to_input
from neural_data_smoothing3D_full.utils import _aver
import h5py

color = ["C" + str(i) for i in range(10)]


def main():
	with resources.path(ASSETS, FILE_NAME) as path:
		print('Reading file', FILE_NAME, '...')
		# data = np.load(path, allow_pickle="TRUE").item()
		with h5py.File(path, 'r') as f:
			data = {key: f[key][()] for key in f.keys()}
	
	n_elem = data["n_elem"] # ["model"]
	L = data["L"]
	radius = data["radius"]
	s = data["s"]
	s_mean = _aver(s)
	dl = data["dl"]
	bend_matrix = data['bend_matrix']
	shear_matrix = data['shear_matrix']
	
	## data point setup
	n_data_pts = 8  # 5 # exlude the initial point at base
	idx_data_pts = np.array(
		[int(100 / (n_data_pts)) * i for i in range(1, n_data_pts)] + [-1]
	)
	print("idx of s_j's:", idx_data_pts)

	position = data["true_pos"]
	director = data["true_dir"]

	input_pos = position[..., idx_data_pts]
	input_dir = director[..., idx_data_pts]
	input_data = pos_dir_to_input(input_pos, input_dir)
	print('# of samples in small data set', len(input_data))

	# true_pos = data['true_pos']
	# true_dir = data['true_dir']
	true_kappa = data["true_kappa"]
	true_shear = data['true_shear']

	n_components = np.array([5 for i in range(6)])
	pca_list = []
	for i in range(len(n_components)):
		pca = PCA(n_components=n_components[i])
		if i < 3:
			pca.fit(true_kappa[:, i, :])
		else:
			pca.fit(true_shear[:, i%3, :])
		pca_list.append(pca)

	flag_save = 1

	if flag_save:
		import os

		folder_name = "assets"
		if not os.path.exists(folder_name):
			os.mkdir(folder_name)

		print("saving data to file", FILE_NAME, '...')

		model_data = {
			"n_elem": n_elem,
			"L": L,
			"radius": radius,
			"s": s,
			"dl": dl,
			'bend_matrix': bend_matrix,
			'shear_matrix': shear_matrix
			}
		for key in model_data.keys():
			data.pop(key, None)
		
		data['model'] = model_data
		data["n_data_pts"] = n_data_pts
		data["idx_data_pts"] = idx_data_pts
		data["input_data"] = input_data
		data["pca"] = pca_list
		np.save(folder_name + '/' + FILE_NAME[:-3], data)

	for i in range(len(n_components)):
		print('strain', i, ': mean =', pca_list[i].mean.mean(), ', std =', pca_list[i].std.std())
		plt.figure(i)
		for j in range(n_components[i]):
			if i < 3:
				plt.plot(s[1:-1], pca_list[i].components[:, j])
			else:
				plt.plot(s_mean, pca_list[i].components[:, j])

	plt.show()


if __name__ == "__main__":
	main()
	# main(sys.argv[1])
