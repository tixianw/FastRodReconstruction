"""
Created on Aug 18, 2024
@author: Tixian Wang
"""
import sys
sys.path.append('../')
import numpy as np
import matplotlib.pyplot as plt
import torch
# from torch import nn, optim
# import torch.nn.functional as F
from time import perf_counter
# from neural_data_smoothing3D import PCA, TensorConstants
from neural_data_smoothing3D import CurvatureSmoothing3DNet
from neural_data_smoothing3D import tensor2numpyVec, coeff2strain, strain2posdir
color = ['C'+str(i) for i in range(10)]
np.random.seed(2024)

def main():

	choice = 0
	folder_name = 'Data/'
	if choice == 0:
		file_name = 'BR2_arm_data'
		model_name = 'data_smoothing_model_br2_BS128'
	elif choice ==1:
		pass
	data = np.load(folder_name + file_name + '.npy', allow_pickle='TRUE').item()
	n_elem = data['model']['n_elem']
	L = data['model']['L']
	radius = data['model']['radius']
	s = data['model']['s']
	dl = data['model']['dl']
	nominal_shear = data['model']['nominal_shear']
	idx_data_pts = data['idx_data_pts']
	pca = data['pca']

	### test data
	test_data_file_name = '../examples/Data/training_data_set'
	training_data = np.load(folder_name + test_data_file_name + '.npy', allow_pickle='TRUE').item()
	input_data = training_data['input_data']
	true_pos = training_data['true_pos']
	true_dir = training_data['true_dir']
	true_kappa = training_data['true_kappa']
	input_size = training_data['input_size']
	output_size = training_data['output_size']

	model = torch.load(folder_name + 'model/'+ model_name +'.pt')
	num_epochs = model['num_epochs']
	batch_size = model['batch_size']
	tensor_constants = model['tensor_constants']
	net = CurvatureSmoothing3DNet(input_size, output_size)
	net.load_state_dict(model['model'])
	losses, vlosses, test_loss = model['losses']
	n_iter = int(len(losses) / len(vlosses))

	print('# epochs:', num_epochs, 'batch size:', batch_size,  'regularizations:', tensor_constants.chi_r, tensor_constants.chi_d)
	print('min_loss:', min(losses))

	plt.figure(0)
	plt.semilogy(np.arange(len(losses)), losses, ls='-', label='train')
	plt.semilogy(np.arange(len(vlosses))*n_iter, vlosses, ls='--', label='validation')
	plt.scatter(len(losses), test_loss, s=50, marker='o', color='C3', label='test')
	plt.xlabel('epochs * iterations')
	plt.ylabel('losses')
	plt.legend()

	# print(input_data.shape, true_kappa.shape, true_pos.shape)
	input_tensor = torch.from_numpy(input_data).float()
	idx_list = np.random.randint(len(input_data), size=10) # [i*10 for i in range(6)]

	# net.eval()
	# print('start...')
	# start = perf_counter()
	# for i in range(len(input_data)):
	# 	output = net(input_tensor[i])
	# 	kappa_output = coeff2strain(tensor2numpyVec(output), pca)
	# 	[position_output, director_output] = strain2posdir(kappa_output, dl)
	# stop = perf_counter()
	# print((stop-start), 1e5/(stop-start))

	net.eval()
	fig = plt.figure(1)
	ax = fig.add_subplot(111, projection='3d')
	fig2, axes = plt.subplots(ncols=3, sharex=True, figsize=(16, 5))
	for ii in range(len(idx_list)):
		i = idx_list[ii]
		output = net(input_tensor[i])
		kappa_output = coeff2strain(tensor2numpyVec(output), pca)
		[position_output, director_output] = strain2posdir(kappa_output, dl, nominal_shear)
		ax.plot(position_output[0,0,:], position_output[0,1,:], position_output[0,2,:], color=color[ii], ls='-')
		ax.plot(true_pos[i,0,:], true_pos[i,1,:], true_pos[i,2,:], color=color[ii], ls='--')
		ax.scatter(input_data[i,0,:], input_data[i,1,:], input_data[i,2,:], s=50, marker='o', color=color[ii])
		ax.set_xlim(-L,0)
		ax.set_ylim(-L,0)
		ax.set_zlim(-L,0)
		ax.set_aspect('equal')
		for j in range(3):
			axes[j].plot(s[1:-1], kappa_output[0,j,:], color=color[ii], ls='-')
			axes[j].plot(s[1:-1], true_kappa[i,j,:], color=color[ii], ls='--')


	plt.show()


if __name__ == "__main__":
	main()