import sys
sys.path.append('../')

import numpy as np
from neural_data_smoothing3D import PCA, TensorConstants # , CurvatureSmoothing3DNet
from reconstruction import ReconstructionModel

def main():
    # Load the model
    model = ReconstructionModel('data_smoothing_model_br2_BS128.pt')

    marker_data = np.zeros(model.input_size)

    # Create a new instance of the model
    result = model(marker_data)

    print(result.position.shape, result.director.shape, result.kappa.shape)

if __name__ == '__main__':
    main()