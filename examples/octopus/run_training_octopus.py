"""
Created on Sep 4, 2024
@author: Tixian Wang
"""

import os
from importlib import resources

import numpy as np

from assets import ASSETS
from file_global import FILE_NAME, N_BASIS
from neural_data_smoothing3D_full import CurvatureSmoothing3DModel, TensorConstants
from neural_data_smoothing3D_full.utils import _aver

# torch.manual_seed(2024)
# npr.seed(2024)


def main():

	folder_name = "assets"
	training_data_name = "training_data_set_octopus_noisy_%dbasis.npy"%N_BASIS
	if not os.path.exists(folder_name):
		raise FileNotFoundError("Run create_training_set_octopus.py first")

	# Construct the full path to the file
	file_path = os.path.join(folder_name, training_data_name)

	# Check if the file exists
	if not os.path.exists(file_path):
		raise FileNotFoundError(
			f"The file '{training_data_name}' does not exist in the '{folder_name}' folder. "
			f"Run create_training_set_octopus.py first"
		)

	with resources.path(ASSETS, FILE_NAME) as path:
		data = np.load(path, allow_pickle="TRUE").item()

	L = data["model"]["L"]
	dl = data["model"]["dl"]
	bend_matrix = data["model"]['bend_matrix']
	shear_matrix = data["model"]['shear_matrix']
	idx_data_pts = data["idx_data_pts"]
	n_data_pts = data['n_data_pts']
	pca = data["pca"]

	training_data = np.load(
		file_path, allow_pickle="TRUE"
	).item()
	input_data = training_data["input_data"]
	true_dir = training_data["true_dir"]
	true_kappa = training_data["true_kappa"]
	true_shear = training_data['true_shear']
	input_size = training_data["input_size"]
	output_size = training_data["output_size"]
	print("input_size:", input_size, "output_size:", output_size)

	chi = 1e4 # 1e5
	chi_d = np.ones(n_data_pts) * chi / 8
	chi_r = np.ones(n_data_pts) * chi / L**2

	tensor_constants = TensorConstants(
		bend_matrix,
		shear_matrix,
		idx_data_pts,
		dl,
		true_dir[0,...,0],
		chi_r,
		chi_d,
		pca,
		input_size,
		output_size,
	)
	## Train the model
	num_epochs = int(100)
	batch_size = 128 # 256 # 128 # 64 # 32 # 100
	print(
		"# total samples:",
		len(input_data),
		"# epochs:",
		num_epochs,
		"batch size:",
		batch_size,
	)
	model = CurvatureSmoothing3DModel(
		tensor_constants,
		input_data,
		num_epochs,
		batch_size=batch_size,
		labels=[true_kappa, true_shear],
	)

	model_name = "/data_smoothing_model_octopus_test_%dbasis"%N_BASIS
	model.model_train(file_name=folder_name+model_name, check_epoch_idx=20)


if __name__ == "__main__":
	main()
