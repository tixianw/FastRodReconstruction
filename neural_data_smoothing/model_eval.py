import numpy as np
import matplotlib.pyplot as plt
import torch
# from torch import nn, optim
# import torch.nn.functional as F
from run_PCA import PCA
from NN_model import CurvatureSmoothing2DNet, TensorConstants
from utils import tensor2numpyVec, curvature2position
color = ['C'+str(i) for i in range(10)]


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
output_data_approx = data['output_data_approx']
pca = data['pca']

training_data = np.load(folder_name + 'training_data_set_1e5.npy', allow_pickle='TRUE').item()
input_data = training_data['training_data']
true_pos = training_data['true_pos']
true_kappa = training_data['true_kappa']

# model_name = 'kappa_l2norm_model1' # 'kappa_mse_model1'
model_name = 'V_smoothing_model_chi1e4' # 'data_smoothing_model_chi1e2' # 
model = torch.load(folder_name + 'model/'+ model_name +'.pt')
num_epochs = model['num_epochs']
tensor_constants = model['tensor_constants']
input_size = len(tensor_constants.idx_data_pts) * 2
output_size = tensor_constants.pca_components.shape[1]
net = CurvatureSmoothing2DNet(input_size, output_size)
net.load_state_dict(model['model'])
losses = model['losses']

print('# epochs:', num_epochs)
print('min_loss:', min(losses))

plt.figure(0)
plt.semilogy(np.arange(len(losses)), losses)

input_tensor = torch.from_numpy(input_data).float()
idx_list = [i*10 for i in range(6)]
for ii in range(len(idx_list)):
	i = idx_list[ii]
	output = net(input_tensor[i])
	kappa_output = pca.approximate(tensor2numpyVec(output))
	position_output = curvature2position(kappa_output)
	true_position = curvature2position(true_kappa[i,:])
	plt.figure(1)
	plt.plot(s[1:-1], kappa_output[:], color=color[ii], ls='-')
	plt.plot(s[1:-1], true_kappa[i,:], color=color[ii], ls='--')
	plt.figure(2)
	plt.plot(position_output[0,:], position_output[1,:], color=color[ii], ls='-')
	plt.plot(true_position[0,:], true_position[1,:], color=color[ii], ls='--')
	plt.scatter(input_data[i,0,:], input_data[i,1,:], s=50, marker='o', color=color[ii])

plt.show()