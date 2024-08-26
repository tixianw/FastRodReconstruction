"""
Created on Aug 18, 2024
@author: Tixian Wang
"""

import numpy as np
import torch
# from tqdm import tqdm
from numba import njit


def tensor2numpyVec(tensor):
    return tensor.detach().numpy()  # .flatten()


# @njit(cache=True)
def pos_dir_to_input(pos: np.ndarray, dir: np.ndarray) -> np.ndarray:
    # input_dir : np.ndarray = dir.reshape(len(dir), -1, dir.shape[-1])
    # inputs : np.ndarray = np.hstack([pos, input_dir])
    inputs: np.ndarray = np.hstack(
        [pos, dir[:, :, 0, :], dir[:, :, -1, :]]
    )  # only take d1 and d3 for dir
    return inputs


def _aver(array):
    return 0.5 * (array[..., 1:] + array[..., :-1])


def _aver_kernel(array):
    blocksize = array.shape[-1]
    output = np.empty(array.shape[:-1] + (blocksize + 1,))
    output[..., :-1] = 0.5 * array
    output[..., 1:] = output[..., 1:] + 0.5 * array
    return output


def coeff2strain(
    coeff, pca
):
    num_strain = len(pca)
    strain = []
    start_coeff_idx = 0
    for i in range(num_strain):
        end_coeff_idx = start_coeff_idx + pca[i].n_components
        strain.append(
            pca[i].approximate(coeff[:, start_coeff_idx:end_coeff_idx])
        )
        start_coeff_idx = end_coeff_idx
    kappa = np.stack(strain[:3], axis=1)
    shear = np.stack(strain[3:], axis=1)
    return [kappa, shear]


def strain2posdir(
    strain, dl
):
    n_sample = len(strain[0])
    n_elem = len(dl)
    position = np.zeros([n_sample, 3, n_elem + 1])
    director = np.zeros([n_sample, 3, 3, n_elem])
    director[:, :, :, 0] = np.diag([1, -1, -1])
    for i in range(n_sample):
        forward_path(
            dl, strain[1][i], strain[0][i], position[i], director[i]
        )  # nominal_shear
    return [position, director]


def coeff2posdir(coeff, pca, dl):
    strain = coeff2strain(coeff, pca)
    posdir = strain2posdir(strain, dl)
    return posdir


@njit(cache=True)
def forward_path(dl, shear, kappa, position_collection, director_collection):
    for i in range(dl.shape[0] - 1):
        next_position(
            director_collection[:, :, i],
            shear[:, i] * dl[i],
            position_collection[:, i : i + 2],
        )
        next_director(kappa[:, i] * dl[i], director_collection[:, :, i : i + 2])
    next_position(
        director_collection[:, :, -1],
        shear[:, -1] * dl[-1],
        position_collection[:, -2:],
    )


@njit(cache=True)
def next_position(director, delta, positions):
    positions[:, 1] = positions[:, 0]
    for index_i in range(3):
        for index_j in range(3):
            positions[index_i, 1] += director[index_j, index_i] * delta[index_j]
    return


@njit(cache=True)
def next_director(axis, directors):
    Rotation = get_rotation_matrix(axis)
    for index_i in range(3):
        for index_j in range(3):
            directors[index_i, index_j, 1] = 0
            for index_k in range(3):
                directors[index_i, index_j, 1] += (
                    Rotation[index_k, index_i] * directors[index_k, index_j, 0]
                )
    return


@njit(cache=True)
def get_rotation_matrix(axis):
    angle = np.sqrt(axis[0] ** 2 + axis[1] ** 2 + axis[2] ** 2)

    axis = axis / (angle + 1e-8)
    K = np.zeros((3, 3))
    K[2, 1] = axis[0]
    K[1, 2] = -axis[0]
    K[0, 2] = axis[1]
    K[2, 0] = -axis[1]
    K[1, 0] = axis[2]
    K[0, 1] = -axis[2]

    K2 = np.zeros((3, 3))
    K2[0, 0] = -(axis[1] * axis[1] + axis[2] * axis[2])
    K2[1, 1] = -(axis[2] * axis[2] + axis[0] * axis[0])
    K2[2, 2] = -(axis[0] * axis[0] + axis[1] * axis[1])
    K2[0, 1] = axis[0] * axis[1]
    K2[1, 0] = axis[0] * axis[1]
    K2[0, 2] = axis[0] * axis[2]
    K2[2, 0] = axis[0] * axis[2]
    K2[1, 2] = axis[1] * axis[2]
    K2[2, 1] = axis[1] * axis[2]

    Rotation = np.sin(angle) * K + (1 - np.cos(angle)) * K2
    Rotation[0, 0] += 1
    Rotation[1, 1] += 1
    Rotation[2, 2] += 1

    return Rotation


### torch version

def coeff2strain_torch(coeff, tensor_constants):
    num_strain = len(tensor_constants.n_components)
    strain = []
    start_coeff_idx = 0
    for j in range(2):
        start_component_idx = 0
        for i in range(3):
            end_coeff_idx = start_coeff_idx + tensor_constants.n_components[i+j*3]
            end_component_idx = start_component_idx + tensor_constants.n_components[i+j*3]
            strain_hat_scaled = torch.einsum(
                "nj,ij->ni",
                coeff[:, start_coeff_idx:end_coeff_idx],
                tensor_constants.pca_components[j][:, start_component_idx:end_component_idx],
            )  ## n is batch size
            strain_hat = (
                strain_hat_scaled * tensor_constants.pca_std[j][i, :]
                + tensor_constants.pca_mean[j][i, :]
            )
            strain.append(strain_hat)
            # print(start_coeff_idx, end_coeff_idx, start_component_idx, end_component_idx)
            start_coeff_idx = end_coeff_idx
            start_component_idx = end_component_idx
    
    kappa = torch.stack(strain[:3], axis=1)
    shear = torch.stack(strain[3:], axis=1)
    return [kappa, shear]


def strain2posdir_torch(strain, tensor_constants):
    pos_dir = forward_path_torch(
        tensor_constants.dl, strain[1], strain[0]
    )
    return [
        pos_dir[0][..., tensor_constants.idx_data_pts],
        pos_dir[1][..., tensor_constants.idx_data_pts],
    ]


def coeff2posdir_torch(coeff, tensor_constants):
    strain_hat = coeff2strain_torch(coeff, tensor_constants)
    pos_dir_j_hat = strain2posdir_torch(strain_hat, tensor_constants)
    return pos_dir_j_hat


# @njit(cache=True)
def forward_path_torch(dl, shear, kappa):
    directors = [
        torch.stack(
            [torch.diag(torch.Tensor([1, -1, -1])) for i in range(len(kappa))],
            axis=0,
        )
    ]
    Rotation = get_rotation_matrix_torch(kappa * _aver(dl))
    for i in range(dl.shape[0] - 1):
        directors.append(
            torch.einsum("nij,njk->nik", directors[-1], Rotation[..., i])
        )
    directors = torch.stack(directors, axis=-1)
    positions = integrate_for_position(directors, shear * dl)
    return [positions, directors]


def integrate_for_position(directors, delta):
    arrays = torch.einsum("nijk,njk->nik", directors, delta)
    positions = torch.cumsum(arrays, dim=-1)
    return positions


# @njit(cache=True)
def get_rotation_matrix_torch(axis):
    n_sample = len(axis)
    n_elem = axis.shape[-1] + 1
    angle = torch.norm(axis, dim=1)
    axis = axis / (angle[:, None] + 1e-16)
    # print(axis.shape, angle.shape)

    K = torch.zeros((n_sample, 3, 3, n_elem - 1))
    K[:, 2, 1, :] = axis[:, 0, :]
    K[:, 1, 2, :] = -axis[:, 0, :]
    K[:, 0, 2, :] = axis[:, 1, :]
    K[:, 2, 0, :] = -axis[:, 1, :]
    K[:, 1, 0, :] = axis[:, 2, :]
    K[:, 0, 1, :] = -axis[:, 2, :]

    K2 = (
        torch.einsum("nik,njk->nijk", axis, axis)
        - torch.eye(3)[None, ..., None]
    )

    Rotation = (
        K * torch.sin(angle)[:, None, None, :]
        + K2 * (1 - torch.cos(angle))[:, None, None, :]
        + torch.eye(3)[None, ..., None]
    )
    return Rotation
