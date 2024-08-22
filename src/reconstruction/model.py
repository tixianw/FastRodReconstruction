from typing import Optional

from importlib import resources

import numpy as np
import torch

from assets import ASSETS, FILE_NAME, MODEL_NAME
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
    ):

        if data_file_name:
            self.data_file_name = data_file_name
            rod_data = np.load(self.data_file_name, allow_pickle="TRUE").item()
        else:
            with resources.path(ASSETS, FILE_NAME) as path:
                rod_data = np.load(path, allow_pickle="TRUE").item()
            self.data_file_name = "package_assets:" + ASSETS + "/" + FILE_NAME

        if model_file_name:
            self.model_file_name = model_file_name
            model_data = torch.load(self.model_file_name)
        else:
            with resources.path(ASSETS, MODEL_NAME) as path:
                model_data = torch.load(path)
            self.model_file_name = "package_assets:" + ASSETS + "/" + MODEL_NAME

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
