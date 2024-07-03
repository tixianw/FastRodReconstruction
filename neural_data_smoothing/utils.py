import numpy as np
import torch

def tensor2numpyVec(tensor):
	return tensor.detach().numpy().flatten()

def curvature2position(curvature, n_elem=100, ds=0.002):
	theta = np.zeros(n_elem)
	position = np.zeros([2, n_elem+1])
	theta[1:] = np.cumsum(curvature, axis=-1) * ds
	position[:, 1:] = np.cumsum(np.vstack([np.cos(theta), np.sin(theta)]), axis=-1) * ds
	return position

def coeff2curvature(coeff, pca):
	return pca.approximate(coeff)

def coeff2position(coeff, pca):
	curvature = coeff2curvature(coeff, pca)
	position = curvature2position(curvature)
	return position

def coeff2curvature_torch(coeff, tensor_constants):
	coeff = torch.flatten(coeff)
	kappa_hat_scaled = torch.mv(tensor_constants.pca_components, coeff)
	kappa_hat = kappa_hat_scaled * tensor_constants.pca_std + tensor_constants.pca_mean
	return kappa_hat

def curvature2position_torch(curvature, tensor_constants):
	theta_hat = torch.cumsum(curvature, dim=0) * tensor_constants.ds
	x_hat = torch.cumsum(torch.cos(theta_hat), dim=0) * tensor_constants.ds
	y_hat = torch.cumsum(torch.sin(theta_hat), dim=0) * tensor_constants.ds
	r_j = torch.hstack((x_hat[tensor_constants.idx_data_pts], y_hat[tensor_constants.idx_data_pts]))
	return r_j

def coeff2position_torch(coeff, tensor_constants):
	kappa_hat = coeff2curvature_torch(coeff, tensor_constants)
	r_j_hat = curvature2position_torch(kappa_hat, tensor_constants)
	return r_j_hat
