from typing import Optional

from importlib import resources

import numpy as np
import torch

from assets import ASSETS, FILE_NAME_BR2, MODEL_NAME_BR2
from neural_data_smoothing3D import (
    CurvatureSmoothing3DNet,
    coeff2strain,
    pos_dir_to_input,
    strain2posdir,
    tensor2numpyVec,
)

from .result import ReconstructionResult


def pose_inv(pose: np.ndarray) -> np.ndarray:
    """
    Invert the pose matrix
    """
    pose_inv = np.identity(4)
    pose_inv[:3, :3] = pose[:3, :3].T
    pose_inv[:3, 3] = -pose[:3, :3].T @ pose[:3, 3]
    return pose_inv


class ReconstructionModel:
    def __init__(
        self,
        data_file_name: Optional[str] = None,
        model_file_name: Optional[str] = None,
        number_of_markers: int = 3,
        base_pose: np.ndarray = np.diag([1, -1, -1, 1]),
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

        self.number_of_markers = number_of_markers
        self.calibrated_pose = np.zeros((4, 4, self.number_of_markers))
        self.input_pose = np.repeat(
            np.expand_dims(np.diag([1.0, -1.0, -1.0, 1.0]), axis=2),
            self.number_of_markers - 1,
            axis=2,
        )

        self.base_pose = base_pose
        self.lab_frame_transformation = np.identity(4)
        self.material_frame_transformation = np.repeat(
            np.expand_dims(np.identity(4), axis=2),
            self.number_of_markers,
            axis=2,
        )

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

    def reconstruct(self, marker_pose: np.ndarray) -> ReconstructionResult:

        self.calibrate_pose(marker_pose)

        # update the result with the new marker data
        input_tensor = torch.from_numpy(self.create_input()).float()
        output = self.net(input_tensor)
        kappa = coeff2strain(tensor2numpyVec(output), self.pca)
        [position, director] = strain2posdir(kappa, self.dl, self.nominal_shear)
        self.result.position = position[0]
        self.result.directors = director[0]
        self.result.kappa = kappa[0]
        return self.result

    # def remove_base_pose_offset(self, marker_pose: np.ndarray) -> np.ndarray:
    #     if marker_pose.ndim == 3:
    #         marker_pose = np.expand_dims(marker_pose, axis=0)
    #     assert marker_pose.ndim == 4, "marker_pose must be 3D or 4D array"

    #     updated_marker_pose = np.zeros(marker_pose.shape)

    #     batch_size = marker_pose.shape[0]
    #     number_of_markers = marker_pose.shape[3]

    #     for b in range(batch_size):

    #         translation_offset = -marker_pose[b, :3, 3, 0]
    #         rotation_offset = (
    #             self.__fixed_base_pose[:3, :3] @ marker_pose[b, :3, :3, 0].T
    #         )

    #         for i in range(number_of_markers):
    #             updated_marker_pose[b, :3, 3, i] = self.__transformation_offset[
    #                 :3, :3
    #             ] @ (translation_offset + marker_pose[b, :3, 3, i])
    #             updated_marker_pose[b, :3, :3, i] = (
    #                 rotation_offset @ marker_pose[b, :3, :3, i]
    #             )
    #             updated_marker_pose[b, :3, 3, i] = updated_marker_pose[
    #                 b, :3, 3, i
    #             ] + (
    #                 updated_marker_pose[b, :3, :3, i]
    #                 @ self.__transformation_offset[:3, 3]
    #                 - updated_marker_pose[b, :3, :3, 0]
    #                 @ self.__transformation_offset[:3, 3]
    #             )

    #     return updated_marker_pose

    def process_calibration(self, marker_pose: np.ndarray) -> bool:
        marker_base_pose = marker_pose[:, :, 0]
        if np.allclose(
            marker_base_pose[:3, 3],
            self.lab_frame_transformation[:3, 3],
            atol=1e-4,
        ):
            return True
        self.lab_frame_transformation[:3, 3] = marker_base_pose[:3, 3]

        # position_offset = np.zeros((3, self.number_of_markers))
        # for i in range(self.number_of_markers):
        #     position_offset[:2, i] = (
        #         marker_pose[:2, 3, i] - marker_pose[:2, 3, 0]
        #     )
        #     position_offset[:, i] = (
        #         marker_pose[:3, :3, i].T @ position_offset[:, i]
        #     )
        #     position_offset[2, i] = 0

        # if hasattr(self, "position_offset"):
        #     if np.allclose(position_offset, self.position_offset, atol=1e-4):
        #         print(self.position_offset[:, 1], position_offset[:, 2])
        #         return True
        #     else:
        #         self.position_offset = position_offset
        #         return False
        # else:
        #     self.position_offset = position_offset
        return False

    def calibrate_pose(self, marker_pose: np.ndarray) -> None:
        for i in range(self.number_of_markers):
            self.calibrated_pose[:, :, i] = (
                pose_inv(self.lab_frame_transformation)
                @ marker_pose[:, :, i]
                @ pose_inv(self.material_frame_transformation[:, :, i])
            )

    def create_input(self) -> np.ndarray:
        self.input_pose[:3, 3, :] = self.calibrated_pose[:3, 3, 1:].copy()
        self.input_pose[:3, :3, :] = np.transpose(
            self.calibrated_pose[:3, :3, 1:], (1, 0, 2)
        )

        return pos_dir_to_input(
            pos=self.input_pose[None, :3, 3, :],
            dir=self.input_pose[None, :3, :3, :],
        )
