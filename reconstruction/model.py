import sys
sys.path.append('../')
import numpy as np
import torch
from neural_data_smoothing3D import CurvatureSmoothing3DNet
from neural_data_smoothing3D import coeff2strain, tensor2numpyVec, strain2posdir
from reconstruction import ReconstructionResult

class ReconstructionModel:
    def __init__(self, file_name: str):
        self.file_name = file_name
        
        # load the model from file
        data_folder = '../neural_data_smoothing3D/Data/'

        rod_data = np.load(data_folder + 'BR2_arm_data.npy', allow_pickle='TRUE').item()
        self.n_elem = rod_data['model']['n_elem']
        # L = rod_data['model']['L']
        # radius = rod_data['model']['radius']
        # s = rod_data['model']['s']
        self.dl = rod_data['model']['dl']
        self.nominal_shear = rod_data['model']['nominal_shear']
        # idx_data_pts = rod_data['idx_data_pts']
        self.pca = rod_data['pca']

        model_data = torch.load(data_folder + 'model/' + file_name)
        self.tensor_constants = model_data['tensor_constants']
        self.input_size = self.tensor_constants.input_size
        self.output_size = self.tensor_constants.output_size
        self.net = CurvatureSmoothing3DNet(self.input_size, self.output_size)
        self.net.load_state_dict(model_data['model'])

        self.number_of_elements = self.n_elem

        self.result = ReconstructionResult(self.number_of_elements)

    def __call__(self, marker_data: np.ndarray) -> ReconstructionResult:
        # update the result with the new marker data
        input_tensor = torch.from_numpy(marker_data).float()
        output = self.net(input_tensor)
        kappa = coeff2strain(tensor2numpyVec(output), self.pca)
        [position, director] = strain2posdir(kappa, self.dl, self.nominal_shear)
        self.result.position = position[0]
        self.result.director = director[0]
        self.result.kappa = kappa[0]
        return self.result
