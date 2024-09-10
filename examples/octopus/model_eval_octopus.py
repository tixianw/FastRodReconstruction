"""
Created on Sep 4, 2024
@author: Tixian Wang
"""

import os
from importlib import resources

import matplotlib.pyplot as plt
import numpy as np
import torch
from time import perf_counter
from tqdm import tqdm
from assets import ASSETS
from file_global import FILE_NAME, MODEL_NAME
from neural_data_smoothing3D_full import (
	CurvatureSmoothing3DNet,
	CurvatureSmoothing3DLoss,
	coeff2strain,
	strain2posdir,
	tensor2numpyVec,
)
from neural_data_smoothing3D_full.utils import _aver

color = ["C" + str(i) for i in range(10)]

## import rod parameters
with resources.path(ASSETS, FILE_NAME) as path:
	data = np.load(path, allow_pickle="TRUE").item()

user_data_flag = True # False # 
folder_name = "assets" # 'Data' # 
if user_data_flag:
	test_data_name = "training_data_set_octopus_noisy_4basis.npy"
	# model_name = 'data_smoothing_model_octopus_test.pt'
	model_name = 'data_smoothing_model_octopus_new_4basis' # _test_4basis' # noise2'
	idx = 100
	model_name += '_epoch%03d'%(idx) + '.pt'
	model_file_path = os.path.join(folder_name, model_name)
	test_data_file_path = os.path.join(folder_name, test_data_name)
	model = torch.load(model_file_path)
	test_data = data
	# test_data = np.load(
	#     test_data_file_path, allow_pickle="TRUE"
	# ).item()
	print('Evalulating user\'s trained model...')
else:
	with resources.path(ASSETS, MODEL_NAME) as path:
		model = torch.load(path)
	test_data = data
	print('No user trained model. Evalulating developer\'s trained model...')

## import rod parameters
n_elem = data["model"]["n_elem"]
L = data["model"]["L"]
radius = data["model"]["radius"]
s = data["model"]["s"]
s_mean = _aver(s)
dl = data["model"]["dl"]
# nominal_shear = data["model"]["nominal_shear"]
idx_data_pts = data["idx_data_pts"]
pca = data["pca"]

## test data
input_data = test_data["input_data"]
true_pos = test_data["true_pos"]
true_dir = test_data["true_dir"]
true_kappa = test_data["true_kappa"]
true_shear = test_data['true_shear']

## load trained model
num_epochs = model["num_epochs"]
batch_size = model["batch_size"]
tensor_constants = model["tensor_constants"]
input_size = tensor_constants.input_size
output_size = tensor_constants.output_size
print("input_size:", input_size, "output_size:", output_size)
net = CurvatureSmoothing3DNet(input_size, output_size)
net.load_state_dict(model["model"])
loss_fn = CurvatureSmoothing3DLoss(tensor_constants)
loss_fn.load_state_dict(model['loss_fn'])
losses, vlosses, test_loss = model["losses"]
n_iter = int(len(losses) / len(vlosses))

print(
	"# epochs:",
	num_epochs,
	"batch size:",
	batch_size,
	# "regularizations: chi_r=",
	# tensor_constants.chi_r,
	# ', chi_d=',
	# tensor_constants.chi_d,
	"min_loss:", min(losses)
)

plt.figure(0)
plt.semilogy(np.arange(len(losses))/n_iter, losses, ls="-", label="train")
plt.semilogy(
	np.arange(len(vlosses)), vlosses, ls="--", label="validation"
)
plt.scatter(len(vlosses), test_loss, s=50, marker="o", color="C3", label="test")
plt.xlabel("epochs") #  * iterations
plt.ylabel("losses")
plt.legend()

# print(input_data.shape, true_kappa.shape, true_pos.shape, true_shear.shape)
input_tensor = torch.from_numpy(input_data).float()
np.random.seed(2024) # 
idx_list = np.random.randint(
	len(input_data), size=10
)  # [i*10 for i in range(6)]

net.eval()
test_loss = []
fig = plt.figure(1)
ax = fig.add_subplot(111, projection="3d")
fig2, axes = plt.subplots(ncols=3, nrows=2, sharex=True, figsize=(16, 5))
for ii in range(len(idx_list)):
	i = idx_list[ii]
	with torch.no_grad():
		output = net(input_tensor[i])
		t_loss = loss_fn(output, input_tensor[i][None,:,:])
		test_loss.append(t_loss)
	strain_output = coeff2strain(tensor2numpyVec(output), pca)
	[position_output, director_output] = strain2posdir(strain_output, dl, true_dir[0,...,0])
	ax.plot(
		position_output[0, 0, :],
		position_output[0, 1, :],
		position_output[0, 2, :],
		color=color[ii],
		ls="-",
	)
	ax.plot(
		true_pos[i, 0, :],
		true_pos[i, 1, :],
		true_pos[i, 2, :],
		color=color[ii],
		ls="--",
	)
	ax.scatter(
		input_data[i, 0, :],
		input_data[i, 1, :],
		input_data[i, 2, :],
		s=50,
		marker="o",
		color=color[ii],
	)
	ax.set_xlim(0, L)
	ax.set_ylim(0, L)
	ax.set_zlim(0, L)
	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_zlabel('z')
	ax.set_aspect("equal")
	for j in range(3):
		axes[0][j].plot(s[1:-1], strain_output[0][0, j, :], color=color[ii], ls="-")
		axes[0][j].plot(s[1:-1], true_kappa[i, j, :], color=color[ii], ls="--")
		axes[1][j].plot(s_mean, strain_output[1][0, j, :], color=color[ii], ls="-")
		axes[1][j].plot(s_mean, true_shear[i, j, :], color=color[ii], ls="--")
		axes[0][j].set_ylabel('$\\kappa_%d$'%(j+1))
		axes[1][j].set_ylabel('$\\nu_%d$'%(j+1))


test_loss = np.array(test_loss)
print('test_loss:', test_loss, 'average:', test_loss.mean())

plt.show()
