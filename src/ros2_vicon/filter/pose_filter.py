import numpy as np


def rotation_matrix(skew_symmetric_matrix: np.ndarray) -> np.ndarray:
    length = skew_symmetric_matrix.shape[-1]

    axis = np.array(
        [
            skew_symmetric_matrix[2, 1],
            skew_symmetric_matrix[0, 2],
            skew_symmetric_matrix[1, 0],
        ]
    )
    angle = np.sqrt(axis[0] ** 2 + axis[1] ** 2 + axis[2] ** 2)
    for i in range(length):
        if angle[i] == 0:
            axis[:, i] = np.zeros(3)
        else:
            axis[:, i] = axis[:, i] / angle[i]

    matrix_K = np.zeros(skew_symmetric_matrix.shape)
    matrix_K[2, 1] = axis[0]
    matrix_K[1, 2] = -axis[0]
    matrix_K[0, 2] = axis[1]
    matrix_K[2, 0] = -axis[1]
    matrix_K[1, 0] = axis[2]
    matrix_K[0, 1] = -axis[2]

    matrix_K2 = np.zeros(skew_symmetric_matrix.shape)
    matrix_K2[0, 0] = -(axis[1] * axis[1] + axis[2] * axis[2])
    matrix_K2[1, 1] = -(axis[2] * axis[2] + axis[0] * axis[0])
    matrix_K2[2, 2] = -(axis[0] * axis[0] + axis[1] * axis[1])
    matrix_K2[0, 1] = axis[0] * axis[1]
    matrix_K2[1, 0] = axis[0] * axis[1]
    matrix_K2[0, 2] = axis[0] * axis[2]
    matrix_K2[2, 0] = axis[0] * axis[2]
    matrix_K2[1, 2] = axis[1] * axis[2]
    matrix_K2[2, 1] = axis[1] * axis[2]

    rotation_matrix = np.zeros(skew_symmetric_matrix.shape)
    for i in range(length):
        rotation_matrix[:, :, i] = (
            np.sin(angle[i]) * matrix_K[:, :, i]
            + (1 - np.cos(angle[i])) * matrix_K2[:, :, i]
        )
    rotation_matrix[0, 0] += 1
    rotation_matrix[1, 1] += 1
    rotation_matrix[2, 2] += 1

    return rotation_matrix


class PoseFilter:
    def __init__(
        self,
        director_filter_gain: np.ndarray,
        position_filter_gain: np.ndarray,
    ) -> None:
        assert (
            director_filter_gain.ndim == position_filter_gain.ndim == 1
        ), "gain must be 1D array with the same length as number of markers"
        assert (
            director_filter_gain.shape[0] == position_filter_gain.shape[0]
        ), "number of markers must be same"
        self.number_of_markers = director_filter_gain.shape[0]
        self.director_filter_gain = director_filter_gain
        self.position_filter_gain = position_filter_gain
        self.__filtered_pose = np.zeros((4, 4, self.number_of_markers))
        for i in range(self.number_of_markers):
            self.__filtered_pose[:, :, i] = np.eye(4)

    def update(self, pose: np.ndarray) -> None:
        self.update_director(pose[:3, :3])
        self.update_position(pose[:3, 3])

    def update_director(self, director: np.ndarray) -> None:
        director_error = self.skew_symmetric_projection(
            np.einsum(
                "jin, jkn -> ikn",
                self.__filtered_pose[:3, :3],
                director,
            )
        )
        self.__filtered_pose[:3, :3] = np.einsum(
            "ikn, kjn -> ijn",
            self.__filtered_pose[:3, :3],
            rotation_matrix(
                np.einsum(
                    "n, ijn -> ijn",
                    self.director_filter_gain,
                    director_error,
                )
            ),
        )

    def update_position(self, position: np.ndarray) -> None:
        position_error = position - self.__filtered_pose[:3, 3]
        self.__filtered_pose[:3, 3] += np.einsum(
            "n, in -> in",
            self.position_filter_gain,
            position_error,
        )

    @property
    def pose(self) -> np.ndarray:
        return self.__filtered_pose

    def skew_symmetric_projection(self, matrix: np.ndarray) -> np.ndarray:
        return 0.5 * (matrix - np.transpose(matrix, (1, 0, 2)))
