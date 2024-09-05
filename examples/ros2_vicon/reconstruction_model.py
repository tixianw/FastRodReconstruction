import matplotlib.pyplot as plt
import numpy as np

from neural_data_smoothing3D import pos_dir_to_input
from reconstruction import ReconstructionModel


def main():
    # Load the model
    model = ReconstructionModel()

    input_ = np.array(
        [
            [
                [0.585334, -0.4472669],
                [-0.35500407, -0.75479233],
                [0.72894186, 0.47983423],
                [0.04491347, 0.12998894],
            ],
            [
                [0.04207713, 0.3797205],
                [-0.884537, -0.6459855],
                [-0.46456847, -0.6622047],
                [0.013555, 0.02342232],
            ],
            [
                [0.8096997, 0.80979294],
                [0.30259952, -0.11397935],
                [-0.502812, 0.5755381],
                [-0.07761133, -0.06592969],
            ],
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [1.0, 1.0]],
        ]
    )
    print(input_.shape)
    marker_data = pos_dir_to_input(
        pos=input_[None, :3, 3, :],
        dir=input_[None, :3, :3, :],
    )
    print(marker_data.shape)
    # Create a new instance of the model
    result = model.reconstruct(marker_data)

    print(result.position.shape, result.directors.shape, result.kappa.shape)

    L = 0.18
    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(
        result.position[0, :],
        result.position[1, :],
        result.position[2, :],
        ls="-",
    )
    ax.scatter(
        marker_data[0, 0, :],
        marker_data[0, 1, :],
        marker_data[0, 2, :],
        s=50,
        marker="o",
    )
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_zlim(-L, 0)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    plt.show()


if __name__ == "__main__":
    main()
