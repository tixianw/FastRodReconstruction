"""
Created on Jun 11, 2024
@author: Tixian Wang
"""

import numpy as np
from numpy.linalg import eig
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
from tqdm import tqdm
from create_data_set import PCA
color = ['C'+str(i) for i in range(10)]

class CurvatureSmoothing2DLoss(nn.Module):
    def __init__(self, EI, s, idx_data_pts, chi_u, chi_r, pca_components, pca_mean):
        super().__init__()
        self.EI = torch.from_numpy(EI).float()
        self.s = s
        self.ds = s[1] - s[0]
        self.idx_data_pts = torch.from_numpy(idx_data_pts) # .long().unsqueeze(1)
        self.chi_u = chi_u
        self.chi_r = chi_r
        self.components = torch.from_numpy(pca_components).float()
        self.mean = torch.from_numpy(pca_mean).float()
    
    def potential_energy(self):
        V = 0.5 * torch.sum(self.EI * self.kappa_hat*self.kappa_hat) * self.ds
        return V
        
    def derivative_penalty(self, output):
        components_s = torch.diff(self.components, dim=0) / self.ds
        kappa_hat_s = torch.mv(components_s, output)
        E_u = 0.5 * self.chi_u * torch.sum(kappa_hat_s * kappa_hat_s) * self.ds
        return E_u

    def get_position(self):
        theta_hat = torch.cumsum(self.kappa_hat, dim=0) * self.ds
        x_hat = torch.cumsum(torch.cos(theta_hat), dim=0) * self.ds
        y_hat = torch.cumsum(torch.sin(theta_hat), dim=0) * self.ds
        r_j = torch.hstack((x_hat[self.idx_data_pts], y_hat[self.idx_data_pts]))
        return r_j

    def data_matching_cost(self, input_data):
        position_difference = input_data - self.get_position()
        Phi = 0.5 * self.chi_r * position_difference.pow(2).sum()
        return Phi

    def forward(self, input, output):
        '''
            input_data: discrete position data points
            output: weights of PCA compunents for approximated target curvature
        '''
        input = torch.flatten(input)
        output = torch.flatten(output)
        self.kappa_hat = torch.mv(self.components, output)
        J = self.potential_energy() 
        # print('V:', J, 'Phi:', self.data_matching_cost(input))
        # J += self.derivative_penalty(output) 
        J += self.data_matching_cost(input)
        return J

class CurvatureSmoothing2DNet(nn.Module):
	def __init__(self, input_size, output_size):
		super(CurvatureSmoothing2DNet, self).__init__()
		self.input_size = input_size
		self.output_size = output_size
		self.linear_relu_stack = nn.Sequential(
			nn.Linear(input_size, 128),
			nn.ReLU(),
			nn.Linear(128, 64),
			nn.ReLU(),
			nn.Linear(64, output_size),
		)

	def forward(self, x):
		x = x.view(-1, self.input_size)
		x = self.linear_relu_stack(x)
		return x
    
def CurvatureSmoothing2DModel(EI, s, idx_data_pts, chi_u, chi_r, pca_components, pca_mean):
    input_size = len(idx_data_pts) * 2
    output_size = pca_components.shape[1]
    net = CurvatureSmoothing2DNet(input_size, output_size)
    loss_fn = CurvatureSmoothing2DLoss(EI, s, idx_data_pts, chi_u, chi_r, pca_components, pca_mean)
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    return net, loss_fn, optimizer

### Train the model
def train_model(input_data, EI, s, idx_data_pts, chi_u, chi_r, pca_components, pca_mean):
    num_epochs = 1
    num_step_check = 20
    training_data = torch.from_numpy(input_data).float()
    net, loss_fn, optimizer = CurvatureSmoothing2DModel(EI, s, idx_data_pts, chi_u, chi_r, pca_components, pca_mean)
    losses = []
    for epoch in range(num_epochs):
        current_loss = 0.0
        for i in tqdm(range(len(training_data))):
            X = Variable(training_data[i], requires_grad=True)
            optimizer.zero_grad()
            output = net(X)
            loss = loss_fn(X, output)
            loss.backward(retain_graph=True)
            optimizer.step()
            current_loss += loss.item()
            if i % num_step_check == 0:
                print(f"Epoch {epoch+1}, Loss: {current_loss / num_step_check}")
                losses.append(current_loss / num_step_check)
                current_loss = 0.0
    return net, losses

def tensor2numpyVec(tensor):
    return tensor.detach().numpy().flatten()


folder_name = 'Data/'
file_name = 'training_data_set'
data = np.load(folder_name + file_name + '.npy', allow_pickle='TRUE').item()

n_elem = data['model']['n_elem']
L = data['model']['L']
radius = data['model']['radius']
s = data['model']['s']
ds = s[1] - s[0]
idx_data_pts = data['idx_data_pts']
input_data = data['input_data']
true_pos = data['true_pos']
true_kappa = data['true_kappa']
output_data_approx = data['output_data_approx']
pca = data['pca']

A = np.pi * (0.5 * (radius[1:]+radius[:-1]))**2
EI = 10**6 * (0.5 * (A[1:]+A[:-1]))**2 / (4*np.pi)
chi_u = 1
chi_r = 1
pca_components = pca.components
pca_mean = pca.mean

## Train the model
net, losses = train_model(input_data, EI, s, idx_data_pts, chi_u, chi_r, pca_components, pca_mean)

plt.figure()
plt.plot(np.arange(len(losses)), losses)

input_tensor = torch.from_numpy(input_data).float()
idx_list = [0,1,2,3,4,5]
plt.figure()
# for i in range(len(output_data)):
for ii in range(len(idx_list)):
    i = idx_list[ii]
    output = net(input_tensor[i])
    kappa_output = pca.approximate(tensor2numpyVec(output))
    plt.plot(s[1:-1], kappa_output[:], color=color[ii], ls='-')
    plt.plot(s[1:-1], true_kappa[i,:], color=color[ii], ls='--')

plt.show()