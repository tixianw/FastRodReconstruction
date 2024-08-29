"""
Created on Aug 28, 2024
@author: Tixian Wang
"""

import numpy as np
import numpy.random as npr
import torch
# from tqdm import tqdm
from numba import njit # , prange

npr.seed(2024)

def tensor2numpyVec(tensor):
    return tensor.detach().numpy()  # .flatten()

def pos_dir_to_noisy_input(pos: np.ndarray, dir: np.ndarray, noise_level=0., L=0.18) -> np.ndarray:
    noisy_pos = pos.copy() + npr.randn(*pos.shape) * noise_level * L
    random_axis = npr.randn(*dir[:,:,0,:].shape) * noise_level * 1.
    Rotation = get_rotation_matrix_numpy(random_axis)
    noisy_dir = np.einsum("nijl,njkl->nikl", dir.copy(), Rotation)
    inputs = np.hstack(
        [noisy_pos, noisy_dir[:, :, 0, :], noisy_dir[:, :, -1, :]]
    )  # only take d1 and d3 for dir
    return inputs

# @njit(cache=True)
# def pos_dir_to_noisy_input(pos: np.ndarray, dir: np.ndarray, noise_level=0., L=0.18) -> np.ndarray:
#     n_sample, _, n_marker = pos.shape
    
#     # Create noisy position
#     noisy_pos = np.zeros_like(pos)
#     for n in prange(n_sample):
#         for i in range(3):
#             for l in range(n_marker):
#                 noisy_pos[n, i, l] = pos[n, i, l] + npr.normal(0, 1) * noise_level * L
    
#     # Create random axis
#     random_axis = np.zeros((n_sample, 3, n_marker))
#     for n in range(n_sample):
#         for j in range(3):
#             for l in range(n_marker):
#                 random_axis[n, j, l] = npr.normal(0, 1) * noise_level / np.sqrt(3)
    
#     # Get rotation matrix
#     Rotation = get_rotation_matrix_numba(random_axis)
    
#     # Create noisy direction
#     noisy_dir = np.zeros_like(dir)
#     for n in range(n_sample):
#         for l in range(n_marker):
#             for i in range(3):
#                 for k in range(3):
#                     noisy_dir[n, i, :, l] += dir[n, :, k, l] * Rotation[n, i, k, l]
    
#     # Combine inputs
#     inputs = np.zeros((n_sample, 9, n_marker))
#     for n in range(n_sample):
#         for l in range(n_marker):
#             inputs[n, :3, l] = noisy_pos[n, :, l]
#             inputs[n, 3:6, l] = noisy_dir[n, :, 0, l]
#             inputs[n, 6:9, l] = noisy_dir[n, :, -1, l]
    
#     # idx = 0
#     # print('pos', pos[idx, :, 0], noisy_pos[idx, :, 0], '\n')
#     # print('d1', dir[idx, :, 0, 0], noisy_dir[idx, :, 0, 0], '\n')
#     # print('d3', dir[idx, :, -1, 0], noisy_dir[idx, :, -1, 0], '\n')
#     return inputs

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
):  # , nominal_shear=np.vstack([np.zeros([2,100]), np.ones(100)])):
    num_strain = len(pca)
    strain = []
    start_coeff_idx = 0
    for i in range(num_strain):
        end_coeff_idx = start_coeff_idx + pca[i].n_components
        strain.append(
            pca[i].approximate(coeff[:, start_coeff_idx:end_coeff_idx])
        )
        start_coeff_idx = end_coeff_idx
    kappa = np.stack(strain, axis=1)
    # shear = np.stack([nominal_shear for i in range(len(kappa))])
    # if num_strain == 6:
    # shear = np.hstack(strain[3:])
    return kappa  # [kappa, shear]


def strain2posdir(
    strain, dl, nominal_shear
):  # np.vstack([np.zeros([2,100]), np.ones(100)])
    n_sample = len(strain)  # len(strain[0])
    n_elem = len(dl)
    position = np.zeros([n_sample, 3, n_elem + 1])
    director = np.zeros([n_sample, 3, 3, n_elem])
    director[:, :, :, 0] = np.diag([1, -1, -1])
    for i in range(n_sample):
        forward_path(
            dl, nominal_shear, strain[i], position[i], director[i]
        )  # strain[1][i]
    return [position, director]


def coeff2posdir(coeff, pca, dl, nominal_shear):
    strain = coeff2strain(coeff, pca)
    posdir = strain2posdir(strain, dl, nominal_shear)
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

    axis = axis / (angle + 1e-16)
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

# @njit(parallel=True)
# def get_rotation_matrix_numba(axis):
#     n_sample = len(axis)
#     n_elem = axis.shape[-1] + 1
#     angle = np.zeros(n_sample)
#     for i in prange(n_sample):
#         angle[i] = np.sqrt(np.sum(axis[i]**2))
    
#     axis_normalized = np.zeros_like(axis)
#     for i in prange(n_sample):
#         for j in range(3):
#             axis_normalized[i, j] = axis[i, j] / (angle[i] + 1e-16)
    
#     K = np.zeros((n_sample, 3, 3, n_elem - 1))
#     for i in prange(n_sample):
#         for j in range(n_elem - 1):
#             K[i, 2, 1, j] = axis_normalized[i, 0, j]
#             K[i, 1, 2, j] = -axis_normalized[i, 0, j]
#             K[i, 0, 2, j] = axis_normalized[i, 1, j]
#             K[i, 2, 0, j] = -axis_normalized[i, 1, j]
#             K[i, 1, 0, j] = axis_normalized[i, 2, j]
#             K[i, 0, 1, j] = -axis_normalized[i, 2, j]
    
#     K2 = np.zeros((n_sample, 3, 3, n_elem - 1))
#     for i in prange(n_sample):
#         for j in range(3):
#             for k in range(3):
#                 for l in range(n_elem - 1):
#                     K2[i, j, k, l] = axis_normalized[i, j, l] * axis_normalized[i, k, l]
#                     if j == k:
#                         K2[i, j, k, l] -= 1.0
    
#     Rotation = np.zeros((n_sample, 3, 3, n_elem - 1))
#     for i in prange(n_sample):
#         sin_angle = np.sin(angle[i])
#         cos_angle = np.cos(angle[i])
#         for j in range(3):
#             for k in range(3):
#                 for l in range(n_elem - 1):
#                     Rotation[i, j, k, l] = (
#                         K[i, j, k, l] * sin_angle +
#                         K2[i, j, k, l] * (1 - cos_angle)
#                     )
#                     if j == k:
#                         Rotation[i, j, k, l] += 1.0
    
#     return Rotation


def get_rotation_matrix_numpy(axis):
    n_sample = len(axis)
    n_elem = axis.shape[-1] + 1
    angle = np.linalg.norm(axis, axis=1)
    axis = axis / (angle[:, None] + 1e-16)

    K = np.zeros((n_sample, 3, 3, n_elem - 1))
    K[:, 2, 1, :] = axis[:, 0, :]
    K[:, 1, 2, :] = -axis[:, 0, :]
    K[:, 0, 2, :] = axis[:, 1, :]
    K[:, 2, 0, :] = -axis[:, 1, :]
    K[:, 1, 0, :] = axis[:, 2, :]
    K[:, 0, 1, :] = -axis[:, 2, :]

    K2 = (
        np.einsum("nik,njk->nijk", axis, axis)
        - np.eye(3)[None, ..., None]
    )

    Rotation = (
        K * np.sin(angle)[:, None, None, :]
        + K2 * (1 - np.cos(angle))[:, None, None, :]
        + np.eye(3)[None, ..., None]
    )
    return Rotation


### torch version


def coeff2strain_torch(coeff, tensor_constants):
    num_strain = len(tensor_constants.n_components)
    strain = []
    start_coeff_idx = 0
    for i in range(num_strain):
        end_coeff_idx = start_coeff_idx + tensor_constants.n_components[i]
        kappa_hat_scaled = torch.einsum(
            "nj,ij->ni",
            coeff[:, start_coeff_idx:end_coeff_idx],
            tensor_constants.pca_components[:, start_coeff_idx:end_coeff_idx],
        )  ## n is batch size
        kappa_hat = (
            kappa_hat_scaled * tensor_constants.pca_std[i, :]
            + tensor_constants.pca_mean[i, :]
        )
        strain.append(kappa_hat)
        start_coeff_idx = end_coeff_idx
    return torch.stack(strain, axis=1)


def strain2posdir_torch(strain, tensor_constants):
    pos_dir = forward_path_torch(
        tensor_constants.dl, tensor_constants.nominal_shear, strain
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
    arrays = torch.einsum("nijk,jk->nik", directors, delta)
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
