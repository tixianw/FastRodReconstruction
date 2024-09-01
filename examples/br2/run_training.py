"""
Created on Aug 21, 2024
@author: Tixian Wang
"""

import os
from importlib import resources

import numpy as np

from assets import ASSETS, FILE_NAME_BR2
from neural_data_smoothing3D import CurvatureSmoothing3DModel, TensorConstants
from neural_data_smoothing3D.utils import _aver

# torch.manual_seed(2024)
# npr.seed(2024)


def main():

    folder_name = "assets"
    training_data_name = "training_data_set_br2_exp.npy"
    if not os.path.exists(folder_name):
        raise FileNotFoundError("Run create_training_set.py first")

    # Construct the full path to the file
    file_path = os.path.join(folder_name, training_data_name)

    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"The file '{training_data_name}' does not exist in the '{folder_name}' folder. "
            f"Run create_training_set.py first"
        )

    with resources.path(ASSETS, FILE_NAME_BR2) as path:
        data = np.load(path, allow_pickle="TRUE").item()

    n_elem = data["model"]["n_elem"]
    L = data["model"]["L"]
    radius = data["model"]["radius"]
    s = data["model"]["s"]
    dl = data["model"]["dl"]
    nominal_shear = data["model"]["nominal_shear"]
    idx_data_pts = data["idx_data_pts"]
    # input_data = data["input_data"]
    # true_pos = data["true_pos"]
    # true_dir = data["true_dir"]
    # true_kappa = data["true_kappa"]
    pca = data["pca"]

    training_data = np.load(
        file_path, allow_pickle="TRUE"
    ).item()
    input_data = training_data["input_data"]
    true_pos = training_data["true_pos"]
    true_dir = training_data["true_dir"]
    true_kappa = training_data["true_kappa"]
    # nominal_shear = training_data['true_shear']
    input_size = training_data["input_size"]
    output_size = training_data["output_size"]

    E = 10**6
    G = E * 2 / 3
    A = np.pi * (radius.mean(axis=0)) ** 2
    bend_twist_stiff = ((_aver(A)) ** 2 / (4 * np.pi))[None, None, :] * np.diag(
        [E, E, 2 * G]
    )[..., None]

    power_chi_r = np.array([5, 5]) # 6 # 5 # 4 # 3
    power_chi_d = np.array([5, 5]) # 5 # 6
    chi_r = 10**power_chi_r  # 1
    chi_d = 10**power_chi_d
    chi_u = 0  # 1e-5

    tensor_constants = TensorConstants(
        bend_twist_stiff,
        idx_data_pts,
        dl,
        chi_r,
        chi_d,
        pca,
        input_size,
        output_size,
    )
    ## Train the model
    num_epochs = int(100)
    batch_size = 128 # 256 # 128 # 100
    print(
        "# total samples:",
        len(input_data),
        "# epochs:",
        num_epochs,
        "batch size:",
        batch_size,
        "regularizations:",
        tensor_constants.chi_r,
        tensor_constants.chi_d,
    )
    model = CurvatureSmoothing3DModel(
        tensor_constants,
        input_data,
        num_epochs,
        batch_size=batch_size,
        labels=true_kappa,
    )

    model_name = "/data_smoothing_model_br2_test1"
    model.model_train(file_name=folder_name+model_name, check_epoch_idx=20)

    # flag_save = True

    # if flag_save:
        # model.model_save(folder_name + model_name)


if __name__ == "__main__":
    main()
