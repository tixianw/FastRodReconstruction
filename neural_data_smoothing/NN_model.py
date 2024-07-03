"""
Created on Jun 14, 2024
@author: Tixian Wang
"""

import sys
import numpy as np
import numpy.random as npr
import torch
from torch import nn, optim
import torch.nn.functional as F
from run_PCA import PCA
from utils import coeff2curvature_torch, coeff2position_torch
# from numpy.linalg import eig
# import matplotlib.pyplot as plt
# from tqdm import tqdm

class TensorConstants:
	def __init__(self, EI, idx_data_pts, ds, chi_r, pca):
		self.EI = torch.from_numpy(EI).float()
		self.idx_data_pts = torch.from_numpy(idx_data_pts)
		self.ds = ds
		self.chi_r = chi_r
		self.pca_components = torch.from_numpy(pca.components).float()
		self.pca_mean = torch.from_numpy(pca.mean).float()
		self.pca_std = torch.from_numpy(pca.std).float()

class CustomLoss(nn.Module):
	def __init__(self, tensor_constants):
		super(CustomLoss, self).__init__()
		self.tensor_constants = tensor_constants

	def forward(self, output, targets):
		output = torch.flatten(output)
		targets = torch.from_numpy(targets).float()
		kappa_hat = torch.mv(self.tensor_constants.pca_components, output)
		# loss = 0.5 * (kappa_hat - targets) * (kappa_hat - targets)
		# return loss.mean()
		diff = kappa_hat - targets
		loss = 0.5 * torch.sum(self.tensor_constants.EI * diff*diff) * self.tensor_constants.ds
		return loss

class CurvatureSmoothing2DLoss(nn.Module):
	def __init__(self, tensor_constants):
		super().__init__()
		self.tensor_constants = tensor_constants
	
	def potential_energy(self):
		V = 0.5 * torch.sum(self.tensor_constants.EI * self.kappa_hat*self.kappa_hat) * self.tensor_constants.ds
		return V
		
	# def derivative_penalty(self, output):
	# 	components_s = torch.diff(self.tensor_constants.pca_components, dim=0) / self.tensor_constants.ds
	# 	kappa_hat_s = torch.mv(components_s, output)
	# 	E_u = 0.5 * self.tensor_constants.chi_u * torch.sum(kappa_hat_s * kappa_hat_s) * self.tensor_constants.ds
	# 	return E_u

	# def get_position(self):
	# 	theta_hat = torch.cumsum(self.kappa_hat, dim=0) * self.tensor_constants.ds
	# 	x_hat = torch.cumsum(torch.cos(theta_hat), dim=0) * self.tensor_constants.ds
	# 	y_hat = torch.cumsum(torch.sin(theta_hat), dim=0) * self.tensor_constants.ds
	# 	r_j = torch.hstack((x_hat[self.tensor_constants.idx_data_pts], y_hat[self.tensor_constants.idx_data_pts]))
	# 	return r_j

	def data_matching_cost(self, input, output):
		position_difference = torch.flatten(input) - coeff2position_torch(output, self.tensor_constants) # self.get_position()
		Phi = 0.5 * self.tensor_constants.chi_r * position_difference.pow(2).sum()
		return Phi

	def forward(self, input, output):
		'''
			input_data: discrete position data points
			output: weights of PCA compunents for approximated target curvature
		'''
		# input = torch.flatten(input)
		# output = torch.flatten(output)
		# self.kappa_hat = torch.mv(self.tensor_constants.pca_components, output)
		self.kappa_hat = coeff2curvature_torch(output, self.tensor_constants)
		J = self.potential_energy() 
		# print('V:', J, 'Phi:', self.data_matching_cost(input))
		# J += self.derivative_penalty(output) 
		J += self.data_matching_cost(input, output)
		# J = self.data_matching_cost(input, output)
		return J

class CurvatureSmoothing2DNet(nn.Module):
	def __init__(self, input_size, output_size):
		super(CurvatureSmoothing2DNet, self).__init__()
		self.input_size = input_size
		self.output_size = output_size
		self.linear_relu_stack = nn.Sequential(
			nn.Linear(input_size, 512),
			nn.ReLU(),
			nn.Linear(512, 256),
			nn.ReLU(),
			nn.Linear(256, 128),
			nn.ReLU(),
			nn.Linear(128, 64),
			nn.ReLU(),
			nn.Linear(64, output_size),
		)

	def forward(self, x):
		x = x.view(-1, self.input_size)
		x = self.linear_relu_stack(x)
		return x
	
def CurvatureSmoothing2DModel(tensor_constants):
	input_size = len(tensor_constants.idx_data_pts) * 2
	output_size = tensor_constants.pca_components.shape[1]
	net = CurvatureSmoothing2DNet(input_size, output_size)
	loss_fn = CurvatureSmoothing2DLoss(tensor_constants) # CustomLoss(tensor_constants) # 
	optimizer = optim.Adam(net.parameters(), lr=0.001)
	return net, loss_fn, optimizer

### Train the model
def train_model(input_data, tensor_constants, num_epochs=1, num_step_check=None, target=None):
	if num_step_check is None:
		num_step_check = len(input_data)
	# training_data = torch.from_numpy(input_data).float()
	net, loss_fn, optimizer = CurvatureSmoothing2DModel(tensor_constants)
	losses = []
	current_loss = 0.0
	for epoch in range(num_epochs):
		# random_indices = npr.randint(100000, size=100)
		for j in range(num_step_check):
			i = num_step_check*epoch + j # random_indices[j] # 
			## NN weights use float32 dtype
			X = torch.tensor(input_data[i], dtype=torch.float32, requires_grad=True) # training_data[i]
			output = net(X)
			loss = loss_fn(X, output) # loss_fn(output, target[i]) # 
			optimizer.zero_grad()
			loss.backward() # (retain_graph=True)
			optimizer.step()
			current_loss += loss.item()

		# if i % num_step_check == 0:
		print(f"Epoch {epoch+1}/{num_epochs}, Loss: {current_loss / num_step_check}")
		losses.append(current_loss / num_step_check)
		current_loss = 0.0

	return net, losses, optimizer

def main(model_name):
	# print('Cuda is available?', torch.cuda.is_available())
	# # torch.cuda.device_count()
	# # torch.cuda.current_device()
	print('data file name:', model_name)

	folder_name = 'Data/'
	file_name = 'original_data_set'
	data = np.load(folder_name + file_name + '.npy', allow_pickle='TRUE').item()

	n_elem = data['model']['n_elem']
	L = data['model']['L']
	radius = data['model']['radius']
	s = data['model']['s']
	ds = s[1] - s[0]
	idx_data_pts = data['idx_data_pts']
	# input_data = data['input_data']
	# true_pos = data['true_pos']
	# true_kappa = data['true_kappa']
	# output_data_approx = data['output_data_approx']
	pca = data['pca']

	training_data = np.load(folder_name + 'training_data_set_1e5.npy', allow_pickle='TRUE').item()
	input_data = training_data['training_data']
	true_pos = training_data['true_pos']
	true_kappa = training_data['true_kappa']

	A = np.pi * (0.5 * (radius[1:]+radius[:-1]))**2
	EI = 10**6 * (0.5 * (A[1:]+A[:-1]))**2 / (4*np.pi)
	power_chi_r = 4 # 5 # 4 # 3
	chi_r = 10**power_chi_r # 1
	# chi_u = 1

	tensor_constants = TensorConstants(EI, idx_data_pts, ds, chi_r, pca)
	## Train the model
	print('# training samples:', len(input_data))
	num_epochs = int(1e3)
	num_step_check = 100
	net, losses, optimizer = train_model(input_data, tensor_constants, num_epochs=num_epochs, num_step_check=num_step_check, target=true_kappa)

	flag_save = True
	# model_name = 'kappa_l2norm_model1' # 'kappa_mse_model1'
	# model_name = 'data_smoothing_model_chi1e'+str(power_chi_r)

	if flag_save:
		torch.save({
			'num_epochs': num_epochs,
			'tensor_constants': tensor_constants,
			'model': net.state_dict(),
			'optimizer': optimizer.state_dict(),
			'losses': losses,
			}, 'Data/model/'+model_name+'.pt'
		)

if __name__ == "__main__":
    main(sys.argv[1])