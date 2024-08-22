"""
Created on Aug 21, 2024
@author: Tixian Wang
"""
import sys
sys.path.append('../')
import numpy as np
import numpy.random as npr
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, Subset, random_split
import torch.nn.functional as F
from neural_data_smoothing3D import PCA, TensorConstants, CurvatureSmoothing3DModel
from neural_data_smoothing3D.utils import _aver

# torch.manual_seed(2024)
# npr.seed(2024)

def main():
	folder_name = '../neural_data_smoothing3D/Data/'
	file_name = 'BR2_arm_data'
	data = np.load(folder_name + file_name + '.npy', allow_pickle='TRUE').item()

	n_elem = data['model']['n_elem']
	L = data['model']['L']
	radius = data['model']['radius']
	s = data['model']['s']
	dl = data['model']['dl']
	nominal_shear = data['model']['nominal_shear']
	idx_data_pts = data['idx_data_pts']
	input_data = data['input_data']
	true_pos = data['true_pos']
	true_dir = data['true_dir']
	true_kappa = data['true_kappa']
	pca = data['pca']

	training_data = np.load('Data/' + 'training_data_set2.npy', allow_pickle='TRUE').item() # training_data_set1
	input_data = training_data['input_data']
	true_pos = training_data['true_pos']
	true_dir = training_data['true_dir']
	true_kappa = training_data['true_kappa']
	# nominal_shear = training_data['true_shear']
	input_size = training_data['input_size']
	output_size = training_data['output_size']

	E = 10**6
	G = E * 2/3
	A = np.pi * (radius.mean(axis=0))**2
	bend_twist_stiff = ((_aver(A))**2 / (4*np.pi))[None, None, :] * np.diag([E, E, 2*G])[..., None]

	power_chi_r = 5 # 6 # 5 # 4 # 3
	power_chi_d = 5
	chi_r = 10**power_chi_r # 1
	chi_d = 10**power_chi_d
	chi_u = 0 # 1e-5

	tensor_constants = TensorConstants(bend_twist_stiff, idx_data_pts, dl, chi_r, chi_d, pca, input_size, output_size)
	## Train the model
	num_epochs = int(10)
	batch_size = 128 # 128 # 100
	print('# total samples:', len(input_data), '# epochs:', num_epochs, 'batch size:', batch_size)
	model = CurvatureSmoothing3DModel(tensor_constants, input_data, num_epochs, batch_size=batch_size, labels=true_kappa)

	model.model_train()

	flag_save = True
	model_name = 'data_smoothing_model_br2_4marker'

	if flag_save:
		model.model_save('Data/Model/' + model_name + '.pt')

if __name__ == '__main__':
	main()