import sys

sys.path.append("../")

import numpy as np

from neural_data_smoothing3D import (  # , CurvatureSmoothing3DNet
    PCA,
    TensorConstants,
)
from reconstruction import ReconstructionModel


def main():
    # Load the model
    model = ReconstructionModel()

    marker_data = np.zeros(model.input_size)

    # Create a new instance of the model
    result = model(marker_data)

    print(result.position.shape, result.directors.shape, result.kappa.shape)


if __name__ == "__main__":
    main()
