"""
Created on Jul 17, 2024
@author: Tixian Wang
"""
import numpy as np
import torch

def tensor2numpyVec(tensor):
	return tensor.detach().numpy() # .flatten()

def curvature2position(curvature, n_elem=100, ds=0.002, L=0.2):
	ratio = L / 0.2
	theta = np.zeros(curvature.shape[:-1] + (n_elem,))
	position = np.zeros(curvature.shape[:-1] + (2, n_elem + 1, ))
	theta[..., 1:] = np.cumsum(curvature, axis=-1) * ds
	position[..., 1:] = np.cumsum(np.stack([np.cos(theta), np.sin(theta)], axis=1), axis=-1) * ds
	# theta = np.cumsum(curvature, axis=-1) * ds
	# position = np.cumsum(np.stack([np.cos(theta), np.sin(theta)], axis=1), axis=-1) * ds
	return position * ratio

def coeff2curvature(coeff, pca, ds=0.002):
	# decision = pca.approximate(coeff[..., 1:])
	# kappa = np.cumsum(decision, axis=-1) * ds + coeff[..., [0]]
	# return kappa
	return pca.approximate(coeff)

def coeff2position(coeff, pca):
	curvature = coeff2curvature(coeff, pca)
	position = curvature2position(curvature)
	return position

def coeff2curvature_torch(coeff, tensor_constants):
	# decision_scaled = torch.einsum('nj,ij->ni', coeff[..., 1:], tensor_constants.pca_components) ## n is batch size
	# decision = decision_scaled * tensor_constants.pca_std + tensor_constants.pca_mean
	# kappa_hat = torch.cumsum(decision, dim=1) * tensor_constants.ds + coeff[..., [0]]
	# return kappa_hat
	kappa_hat_scaled = torch.einsum('nj,ij->ni', coeff, tensor_constants.pca_components) ## n is batch size
	kappa_hat = kappa_hat_scaled * tensor_constants.pca_std + tensor_constants.pca_mean
	return kappa_hat

def curvature2position_torch(curvature, tensor_constants):
	theta_hat = torch.cumsum(curvature, dim=1) * tensor_constants.ds
	x_hat = torch.cumsum(torch.cos(theta_hat), dim=1) * tensor_constants.ds
	y_hat = torch.cumsum(torch.sin(theta_hat), dim=1) * tensor_constants.ds
	r_j = torch.hstack((x_hat[:,tensor_constants.idx_data_pts], y_hat[:,tensor_constants.idx_data_pts]))
	return r_j

def coeff2position_torch(coeff, tensor_constants):
	kappa_hat = coeff2curvature_torch(coeff, tensor_constants)
	r_j_hat = curvature2position_torch(kappa_hat, tensor_constants)
	return r_j_hat

def _aver(array):
	return 0.5 * (array[..., 1:] + array[..., :-1])

def _aver_kernel(array):
	blocksize = array.shape[-1]
	output = np.empty(array.shape[:-1] + (blocksize + 1,))
	output[..., :-1] = 0.5 * array
	output[..., 1:] += 0.5 * array
	return output