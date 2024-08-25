from typing import Optional, Tuple

import numpy as np

from ros2_vicon.message.array import NDArrayDescriptor, NDArrayMessage


class QuaternionDescriptor(NDArrayDescriptor):
    """
    Descriptor for quaternion arrays.
    """

    def __init__(self):
        super().__init__(shape=(4,))

    def __set__(self, obj: object, value: np.ndarray) -> None:
        assert isinstance(
            value, np.ndarray
        ), f"{self.name} must be a numpy array"
        if value.shape != self._shape:
            raise ValueError(f"{self.name} must have shape {self._shape}")
        if not np.isclose(np.linalg.norm(value), 1.0):
            raise ValueError(f"{self.name} must be a unit quaternion")
        setattr(obj, self.private_name, value)


class PoseDescriptor(NDArrayDescriptor):
    """
    Descriptor for pose arrays.
    """

    def __init__(self):
        super().__init__(shape=(4, 4))

    def __set__(self, obj: object, value: np.ndarray) -> None:
        assert isinstance(
            value, np.ndarray
        ), f"{self.name} must be a numpy array"
        if value.shape != self._shape:
            raise ValueError(
                f"{self.name} must have shape {self._shape}, and the value have shape {value.shape}"
            )
        if not np.isclose(np.linalg.det(value[:3, :3]), 1.0):
            raise ValueError(f"{self.name} must be SO(3) rotation matrix")
        if not np.isclose(
            value[
                3,
                3,
            ],
            1.0,
        ):
            raise ValueError(f"{self.name} must be SE(3) transformation matrix")
        setattr(obj, self.private_name, value)


class PoseMessage(NDArrayMessage):
    """
    Class for Pose message data.
    """

    def __init__(self, shape: Tuple[int], axis_labels: Tuple[str]):
        """
        Initialize the PoseMessage object.
        """
        super().__init__(
            shape=(4, 4) + shape, axis_labels=("pose", "") + axis_labels
        )

    @property
    def position(self) -> np.ndarray:
        """
        Return the position vector.
        """
        return self.data[:3, 3, ...]

    @property
    def directors(self) -> np.ndarray:
        """
        Return the directors matrix.
        """
        return self.data[:3, :3, ...]

    def __str__(self) -> str:
        """
        Return the string information of the PoseMessage object.
        """
        return (
            f"\nPoseMessage(\n"
            f"    shape={self.shape},\n"
            f"    axis_labels={self.axis_labels},\n"
            f"    pose=\n{np.array2string(self.data, precision=4, suppress_small=True)},\n"
            f")"
        )
