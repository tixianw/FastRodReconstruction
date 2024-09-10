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
        number_of_markers: int = 4,
        base_pose: np.ndarray = np.diag([1, -1, -1, 1]),
    ):

        self.number_of_markers = number_of_markers
        if data_file_name:
            self.data_file_name = data_file_name
            rod_data = np.load(self.data_file_name, allow_pickle="TRUE").item()
        else:
            with resources.path(
                ASSETS, FILE_NAME_BR2[self.number_of_markers]
            ) as path:
                rod_data = np.load(path, allow_pickle="TRUE").item()
            self.data_file_name = (
                "package_assets:"
                + ASSETS
                + "/"
                + FILE_NAME_BR2[self.number_of_markers]
            )

        if model_file_name:
            self.model_file_name = model_file_name
            model_data = torch.load(self.model_file_name)
        else:
            with resources.path(
                ASSETS, MODEL_NAME_BR2[self.number_of_markers]
            ) as path:
                model_data = torch.load(path)
            self.model_file_name = (
                "package_assets:"
                + ASSETS
                + "/"
                + MODEL_NAME_BR2[self.number_of_markers]
            )

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

        self.n_elem = rod_data["model"]["n_elem"]
        # L = rod_data['model']['L']
        # radius = rod_data['model']['radius']
        # s = rod_data['model']['s']
        self.dl = rod_data["model"]["dl"]
        self.nominal_shear = rod_data["model"]["nominal_shear"]
        self.pca = rod_data["pca"]

        self.tensor_constants = model_data["tensor_constants"]
        self.idx_data_pts = self.tensor_constants.idx_data_pts
        self.input_size = self.tensor_constants.input_size
        self.output_size = self.tensor_constants.output_size
        self.net = CurvatureSmoothing3DNet(self.input_size, self.output_size)
        self.net.load_state_dict(model_data["model"])

        self.number_of_elements = self.n_elem

        self.result = ReconstructionResult(self.number_of_elements)
        self.cost = np.nan

    def __str__(self) -> str:
        return (
            f"ReconstructionModel("
            f"\n    data_file_name={self.data_file_name}, "
            f"\n    model_file_name={self.model_file_name}\n)"
        )

    def reconstruct(self, marker_pose: np.ndarray) -> None:

        # calibrate the pose
        self.calibrate_pose(marker_pose)

        # create the input tensor
        input_tensor = torch.from_numpy(self.create_input()).float()

        # get the output from the neural network
        output = self.net(input_tensor)

        # convert the output to strain, position and director
        kappa = coeff2strain(tensor2numpyVec(output), self.pca)
        [position, director] = strain2posdir(
            kappa, self.dl, self.nominal_shear, np.diag([1.0, -1.0, -1.0])
        )

        # update the result with the new marker data
        self.result.position = position[0]
        self.result.directors = director[0]
        self.result.kappa = kappa[0]

    def process_lab_frame_calibration(self, marker_pose: np.ndarray) -> bool:
        marker_base_pose = marker_pose[:, :, 0]
        if np.allclose(
            marker_base_pose[:3, 3],
            self.lab_frame_transformation[:3, 3],
            atol=1e-4,
        ):
            return True
        self.lab_frame_transformation[:3, 3] = marker_base_pose[:3, 3]
        return False

    def process_material_frame_calibration(
        self, marker_pose: np.ndarray
    ) -> bool:
        return True

    def calibrate_pose(self, marker_pose: np.ndarray) -> None:
        for i in range(self.number_of_markers):
            self.calibrated_pose[:, :, i] = (
                pose_inv(self.lab_frame_transformation)
                @ marker_pose[:, :, i]
                @ pose_inv(self.material_frame_transformation[:, :, i])
            )
        for i in reversed(range(self.number_of_markers)):
            self.calibrated_pose[:3, 3, i] = (
                self.calibrated_pose[:3, 3, i] - self.calibrated_pose[:3, 3, 0]
            )

    def create_input(self) -> np.ndarray:
        self.input_pose[:3, 3, :] = self.calibrated_pose[:3, 3, 1:]
        # The transpose is added because the input pose is
        # expected to be in the format of row vectors due to
        # the setting of directors in pyelastica
        self.input_pose[:3, :3, :] = np.transpose(
            self.calibrated_pose[:3, :3, 1:], (1, 0, 2)
        )

        return pos_dir_to_input(
            pos=self.input_pose[None, :3, 3, :],
            dir=self.input_pose[None, :3, :3, :],
        )
