"""
Created on Jul 17, 2024
@author: Tixian Wang
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from NN_model import CurvatureSmoothing2DNet, TensorConstants

# from torch import nn, optim
# import torch.nn.functional as F
from run_PCA import PCA
from utils import coeff2curvature, curvature2position, tensor2numpyVec

color = ["C" + str(i) for i in range(10)]
np.random.seed(2024)

# folder_name = 'Data/'
# file_name = 'fake_br2_arm_data'
# data = np.load(folder_name + file_name + '.npy', allow_pickle='TRUE').item()
# n_elem = data['model']['n_elem']
# L = data['model']['L']
# radius = data['model']['radius']
# s = data['model']['s']
# ds = s[1] - s[0]
# idx_data_pts = data['idx_data_pts']
# input_data = data['input_data']
# true_pos = data['true_pos']
# true_kappa = data['true_kappa']
# print(data.keys())
# del data['pca']
# del data['output_data_approx']
# print(data.keys())
# np.save('Data/fake_br2_arm_data111.npy', data)
# quit()

choice = 0
folder_name = "Data/"
# model_name = 'kappa_l2norm_model1' # 'kappa_mse_model1'
# model_name = 'V_smoothing_model_chi1e4' # 'data_smoothing_model_chi1e2' # 'small_model_test1'
if choice == 0:
    file_name = "fake_br2_arm_data1"
    model_name = "mini_batch_model_br2"  # 'curvature_derivative_model_br2' # _chi1e-6_1e6' #
elif choice == 1:
    file_name = "pyelastica_arm_data"
    model_name = "mini_batch_model_pyelastica"  # 'curvature_derivative_model_pyelastica' #
data = np.load(folder_name + file_name + ".npy", allow_pickle="TRUE").item()
n_elem = data["model"]["n_elem"]
L = data["model"]["L"]
radius = data["model"]["radius"]
s = data["model"]["s"]
ds = s[1] - s[0]
idx_data_pts = data["idx_data_pts"]
if choice == 1:
    input_data = data["input_data"]
    true_pos = data["true_pos"]
    true_kappa = data["true_kappa"]
pca = data["pca"]

### test data
test_data_file_name = "training_data_set1"  # 'pyelastica_arm_data2' #
training_data = np.load(
    folder_name + test_data_file_name + ".npy", allow_pickle="TRUE"
).item()
input_data = training_data["training_data"]  # ['input_data'] #
true_pos = training_data["true_pos"]
true_kappa = training_data["true_kappa"]


model = torch.load(folder_name + "model/" + model_name + ".pt")
num_epochs = model["num_epochs"]
tensor_constants = model["tensor_constants"]
input_size = len(tensor_constants.idx_data_pts) * 2
output_size = tensor_constants.pca_components.shape[1]  # + 1
net = CurvatureSmoothing2DNet(input_size, output_size)
net.load_state_dict(model["model"])
losses, vlosses, test_loss = model["losses"]
n_iter = int(len(losses) / len(vlosses))

print("# epochs:", num_epochs, "regularization chi_r:", tensor_constants.chi_r)
print("min_loss:", min(losses))

plt.figure(0)
plt.semilogy(np.arange(len(losses)), losses, ls="-", label="train")
plt.semilogy(
    np.arange(len(vlosses)) * n_iter, vlosses, ls="--", label="validation"
)
plt.scatter(len(losses), test_loss, s=50, marker="o", color="C3", label="test")
plt.xlabel("epochs * iterations")
plt.ylabel("losses")
plt.legend()

# print(input_data.shape, true_kappa.shape, true_pos.shape)
input_tensor = torch.from_numpy(input_data).float()
idx_list = np.random.randint(
    len(input_data), size=10
)  # [i*10 for i in range(6)]

net.eval()
for ii in range(len(idx_list)):
    i = idx_list[ii]
    output = net(input_tensor[i])
    kappa_output = coeff2curvature(tensor2numpyVec(output), pca)
    position_output = curvature2position(kappa_output)
    # true_position = curvature2position(true_kappa[i,:])
    plt.figure(1)
    plt.plot(s[1:-1], kappa_output[0, :], color=color[ii], ls="-")
    plt.plot(s[1:-1], true_kappa[i, :], color=color[ii], ls="--")
    plt.figure(2)
    plt.plot(
        position_output[0, 0, :],
        position_output[0, 1, :],
        color=color[ii],
        ls="-",
    )
    plt.plot(true_pos[i, 0, :], true_pos[i, 1, :], color=color[ii], ls="--")
    plt.scatter(
        input_data[i, 0, :],
        input_data[i, 1, :],
        s=50,
        marker="o",
        color=color[ii],
    )
    # plt.axis([-0.5*L, 1.2*L, -0.5*L, 1.2*L])

plt.show()
