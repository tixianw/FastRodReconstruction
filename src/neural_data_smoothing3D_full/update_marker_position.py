"""
Created on Aug 21, 2024
@author: Tixian Wang
"""

from importlib import resources

import numpy as np

from assets import ASSETS
from assets import FILE_NAME_OCTOPUS as FILE_NAME
from neural_data_smoothing3D_full import pos_dir_to_input


def main():
    with resources.path(ASSETS, FILE_NAME) as path:
        print('Reading file', FILE_NAME, '...')
        data = np.load(path, allow_pickle="TRUE").item()

    ## data point setup
    n_data_pts = 9  # 5 # exlude the initial point at base
    idx_data_pts = np.array(
        [int(100 / (n_data_pts)) * i for i in range(1, n_data_pts)] + [-1]
    )
    print("idx of s_j's:", idx_data_pts)

    position = data["true_pos"]
    director = data["true_dir"]

    input_pos = position[..., idx_data_pts]
    input_dir = director[..., idx_data_pts]
    input_data = pos_dir_to_input(input_pos, input_dir)
    # print(position.shape, director.shape, input_pos.shape, input_data.shape)

    flag_save = 0

    if flag_save:
        print("updating data to file", FILE_NAME, '...')

        data["n_data_pts"] = n_data_pts
        data["idx_data_pts"] = idx_data_pts
        data["input_data"] = input_data

        with resources.path(ASSETS, FILE_NAME) as path:
            np.save(path, data)


if __name__ == "__main__":
    main()
