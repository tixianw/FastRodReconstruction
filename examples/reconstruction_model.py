import numpy as np

from reconstruction import ReconstructionModel


def main():
    # Load the model
    model = ReconstructionModel()

    marker_data = np.zeros(model.input_size)
    print(marker_data.shape)

    # Create a new instance of the model
    result = model(marker_data)

    print(result.position.shape, result.directors.shape, result.kappa.shape)


if __name__ == "__main__":
    main()
