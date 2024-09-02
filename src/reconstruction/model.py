from typing import Optional

from importlib import resources

import numpy as np
import torch

from assets import ASSETS, FILE_NAME_BR2, MODEL_NAME_BR2
from neural_data_smoothing3D import (
    CurvatureSmoothing3DNet,
    coeff2strain,
    strain2posdir,
    tensor2numpyVec,
)

from .result import ReconstructionResult


class ReconstructionModel:
    def __init__(
        self,
        data_file_name: Optional[str] = None,
        model_file_name: Optional[str] = None,
        fixed_base_pose: np.ndarray = np.diag([1, -1, -1, 1]),
    ):

        if data_file_name:
            self.data_file_name = data_file_name
            rod_data = np.load(self.data_file_name, allow_pickle="TRUE").item()
        else:
            with resources.path(ASSETS, FILE_NAME_BR2) as path:
                rod_data = np.load(path, allow_pickle="TRUE").item()
            self.data_file_name = (
                "package_assets:" + ASSETS + "/" + FILE_NAME_BR2
            )

        if model_file_name:
            self.model_file_name = model_file_name
            model_data = torch.load(self.model_file_name)
        else:
            with resources.path(ASSETS, MODEL_NAME_BR2) as path:
                model_data = torch.load(path)
            self.model_file_name = (
                "package_assets:" + ASSETS + "/" + MODEL_NAME_BR2
            )

        self.__base_pose = fixed_base_pose.copy()
        self.__fixed_base_pose = fixed_base_pose.copy()
        self.__transformation_offset = np.eye(4)

        # self.data_file_name = data_file_name if data_file_name else ASSETS + "/" + FILE_NAME
        # self.model_file_name = model_file_name if model_file_name else ASSETS + "/" + MODEL_NAME
        # load the model from file
        # data_folder = "../neural_data_smoothing3D/Data/"

        self.n_elem = rod_data["model"]["n_elem"]
        # L = rod_data['model']['L']
        # radius = rod_data['model']['radius']
        # s = rod_data['model']['s']
        self.dl = rod_data["model"]["dl"]
        self.nominal_shear = rod_data["model"]["nominal_shear"]
        # idx_data_pts = rod_data['idx_data_pts']
        self.pca = rod_data["pca"]

        self.tensor_constants = model_data["tensor_constants"]
        self.idx_data_pts = self.tensor_constants.idx_data_pts
        self.input_size = self.tensor_constants.input_size
        self.output_size = self.tensor_constants.output_size
        self.net = CurvatureSmoothing3DNet(self.input_size, self.output_size)
        self.net.load_state_dict(model_data["model"])

        self.number_of_elements = self.n_elem

        self.result = ReconstructionResult(self.number_of_elements)

    def __str__(self) -> str:
        return (
            f"ReconstructionModel("
            f"\n    data_file_name={self.data_file_name}, "
            f"\n    model_file_name={self.model_file_name}\n)"
        )

    def __call__(self, marker_data: np.ndarray) -> ReconstructionResult:
        # update the result with the new marker data
        input_tensor = torch.from_numpy(marker_data).float()
        output = self.net(input_tensor)
        kappa = coeff2strain(tensor2numpyVec(output), self.pca)
        [position, director] = strain2posdir(kappa, self.dl, self.nominal_shear)
        self.result.position = position[0]
        self.result.directors = director[0]
        self.result.kappa = kappa[0]
        return self.result

    def set_base_pose(self, base_pose: np.ndarray) -> None:
        self.__base_pose = base_pose.copy()

    def set_rotation_offset(self, angle: float):
        angle = np.deg2rad(angle)
        self.__transformation_offset[:3, :3] = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1],
            ]
        )

    def set_translation_offset(self, translation_offset: np.ndarray):
        self.__transformation_offset[:3, 3] = translation_offset.copy()

    @property
    def transformation_offset(self) -> np.ndarray:
        return self.__transformation_offset

    # def remove_base_translation(
    #     self, marker_position: np.ndarray
    # ) -> np.ndarray:
    #     updated_marker_position = marker_position.copy()
    #     for i in range(marker_position.shape[0]):
    #         for j in range(marker_position.shape[2]):
    #             updated_marker_position[i, :, j] = (
    #                 self.__rotation_offset
    #                 @ (marker_position[i, :, j] - self.__base_pose[:3, 3])
    #             )
    #     return updated_marker_position

    # def remove_base_rotation(self, marker_directors: np.ndarray) -> np.ndarray:
    #     update_maker_directors = marker_directors.copy()
    #     rotation_matrix = (
    #         self.__rotation_offset @ self.__base_pose[:3, :3]
    #     ).T @ self.__fixed_base_pose[:3, :3]
    #     for i in range(marker_directors.shape[0]):
    #         for j in range(marker_directors.shape[3]):
    #             update_maker_directors[i, :, :, j] = (
    #                 self.__rotation_offset
    #                 @ marker_directors[i, :, :, j]
    #                 @ rotation_matrix
    #             ).T
    #     return update_maker_directors

    def remove_base_pose_offset(self, marker_pose: np.ndarray) -> np.ndarray:
        if marker_pose.ndim == 3:
            marker_pose = np.expand_dims(marker_pose, axis=0)
        assert marker_pose.ndim == 4, "marker_pose must be 3D or 4D array"

        updated_marker_pose = np.zeros(marker_pose.shape)

        batch_size = marker_pose.shape[0]
        number_of_markers = marker_pose.shape[3]

        # translation_offset = -self.__base_pose[:3, 3]
        # rotation_offset = (
        #     self.__fixed_base_pose[:3, :3] @ self.__base_pose[:3, :3].T
        # )

        for b in range(batch_size):

            translation_offset = -marker_pose[b, :3, 3, 0]
            rotation_offset = (
                self.__fixed_base_pose[:3, :3] @ marker_pose[b, :3, :3, 0].T
            )

            for i in range(number_of_markers):
                updated_marker_pose[b, :3, 3, i] = self.__transformation_offset[
                    :3, :3
                ] @ (translation_offset + marker_pose[b, :3, 3, i])
                updated_marker_pose[b, :3, :3, i] = (
                    rotation_offset @ marker_pose[b, :3, :3, i]
                )
                updated_marker_pose[b, :3, 3, i] = updated_marker_pose[
                    b, :3, 3, i
                ] + (
                    updated_marker_pose[b, :3, :3, i]
                    @ self.__transformation_offset[:3, 3]
                    - updated_marker_pose[b, :3, :3, 0]
                    @ self.__transformation_offset[:3, 3]
                )

            # for i in range(number_of_markers):
            #     updated_marker_pose[b, :3, 3, i] = self.__transformation_offset[:3, :3] @ (
            #         translation_offset + marker_pose[b, :3, 3, i]
            #     )
            #     updated_marker_pose[b, :3, :3, i] = (
            #         rotation_offset @ marker_pose[b, :3, :3, i]
            #     )
            #     updated_marker_pose[b, :3, 3, i] = updated_marker_pose[b, :3, 3, i] + (
            #         updated_marker_pose[b, :3, :3, i] @ self.__transformation_offset[:3, 3] -
            #         self.__base_pose[:3, :3] @ self.__transformation_offset[:3, 3]
            #     )

        # Remove offset
        # for i in range(marker_pose.shape[0]):
        #     for j in range(marker_pose.shape[3]):
        #         updated_marker_pose[i, :3, 3, j] = (
        #             marker_pose[i, :3, 3, j] - self.__base_pose[:3, 3]
        #         )

        # Align bending direction
        # for i in range(marker_pose.shape[0]):
        #     for j in range(marker_pose.shape[3]):
        #         updated_marker_pose[i, :3, :, j] = (
        #             self.__bending_rotation_matrix @ updated_marker_pose[i, :3, :, j]
        #         )
        #         updated_marker_pose[i, :3, :3, j] = (
        #             self.__bending_rotation_matrix @ updated_marker_pose[i, :3, :3, j]
        #         )

        # Remove rotation
        # rotation_matrix = (
        #     self.__fixed_base_pose[:3, :3] @ self.__base_pose[:3, :3].T
        # )
        # for i in range(marker_pose.shape[0]):
        #     for j in range(marker_pose.shape[3]):
        #         updated_marker_pose[i, :3, :3, j] = (
        #             rotation_matrix @ updated_marker_pose[i, :3, :3, j]
        #         )
        #         updated_marker_pose[i, :3, 3, j] = (
        #             rotation_matrix @ updated_marker_pose[i, :3, 3, j]
        #         )
        return updated_marker_pose
