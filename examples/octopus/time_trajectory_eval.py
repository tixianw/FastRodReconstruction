"""
Created on Sep 4, 2024
@author: Tixian Wang
"""

import os
from importlib import resources

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
import torch
# from time import perf_counter
import h5py
from assets import ASSETS
from file_global import FILE_NAME, MODEL_NAME
from neural_data_smoothing3D_full import (
    CurvatureSmoothing3DNet,
    CurvatureSmoothing3DLoss,
    coeff2strain,
    strain2posdir,
    tensor2numpyVec,
    pos_dir_to_input
)
from neural_data_smoothing3D_full.utils import _aver

color = ["C" + str(i) for i in range(10)]
np.random.seed(2024)

## import rod parameters
with resources.path(ASSETS, FILE_NAME) as path:
    data = np.load(path, allow_pickle="TRUE").item()

user_data_flag = True # False # 
folder_name = "assets"
if user_data_flag:
    test_data_name = "training_data_set_octopus_noisy.npy " # _6basis.npy" # 
    # model_name = 'data_smoothing_model_octopus_test.pt'
    model_name = 'data_smoothing_model_octopus_test_noise2' # 6basis' # 
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


## import demo simulation data
DEMO_FILE_NAME = "octopus_arm_data_demo.h5"
with resources.path(ASSETS, DEMO_FILE_NAME) as path:
		print('Reading file', DEMO_FILE_NAME, '...')
		with h5py.File(path, 'r') as f:
			demo_data = {key: f[key][()] for key in f.keys()}
## data point setup
true_pos = demo_data['true_pos']
true_dir = demo_data['true_dir']
true_kappa = demo_data["true_kappa"]
true_shear = demo_data['true_shear']
input_pos = true_pos[..., idx_data_pts]
input_dir = true_dir[..., idx_data_pts]
input_data = pos_dir_to_input(input_pos, input_dir)
print('# of samples in small data set', len(input_data))

# ## test data
# input_data = test_data["input_data"]
# true_pos = test_data["true_pos"]
# true_dir = test_data["true_dir"]
# true_kappa = test_data["true_kappa"]
# true_shear = test_data['true_shear']

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


# print(input_data.shape, true_kappa.shape, true_pos.shape, true_shear.shape)
input_tensor = torch.from_numpy(input_data).float()


fig = plt.figure(1)
ax = fig.add_subplot(111, projection="3d")
fig2, axes = plt.subplots(ncols=3, nrows=2, sharex=True, figsize=(16, 5))
fps = 60
factor = 5
file_name = 'octopus_arm_tracking'
video_name = folder_name + '/' + file_name + ".mov"
FFMpegWriter = manimation.writers["ffmpeg"]
metadata = dict(title="Movie Test", artist="Matplotlib", comment="Movie support!")
writer = FFMpegWriter(fps=fps, metadata=metadata)
net.eval()
print('# of test samples:', len(input_data))
start = 0 # 2000 # 0 # 
end = min(start + 1000, len(input_data))
video_save_flag = False
with writer.saving(fig, video_name, 100):
    for k in range(int((end-start)/factor)+1):
        i = start + k*factor
        ii = i % 10
        ax.cla()
        for j in range(2):
            for k in range(3):
                axes[j][k].cla()
        with torch.no_grad():
            output = net(input_tensor[i])
            t_loss = loss_fn(output, input_tensor[i][None,:,:])
        strain_output = coeff2strain(tensor2numpyVec(output), pca)
        [position_output, director_output] = strain2posdir(strain_output, dl, true_dir[0,...,0])
        ax.text(0.05, 1.05, 1.2, 'data idx: %5d'%(i), transform=ax.transAxes, fontsize=12)
        ax.text(0.05, 1.05, 1.0, 'test loss: %.3f'%(t_loss), transform=ax.transAxes, fontsize=12)
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
        
        if not video_save_flag:
            plt.pause(0.001)
        else:
            writer.grab_frame()


plt.show()
