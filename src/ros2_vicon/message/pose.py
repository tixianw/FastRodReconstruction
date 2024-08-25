from typing import Union

import numpy as np
from scipy.spatial.transform import Rotation

from ros2_vicon.message.array import NDArrayDescriptor

try:
    from vicon_receiver.msg import Position as Pose
except ModuleNotFoundError:
    print(
        "Could not import ROS2 modules. Make sure to source ROS2 workspace first."
    )
    import sys

    sys.exit(1)


class QuaternionDescriptor(NDArrayDescriptor):
    """
    Descriptor for quaternion arrays.
    """

    def __init__(
        self,
    ):
        super().__init__(shape=(4,))

    def __set__(self, obj: object, value: Union[np.ndarray, list]) -> None:
        if isinstance(value, list):
            value = np.array(value)
        if not isinstance(value, np.ndarray):
            raise TypeError(f"{self.name} must be a numpy array or list")
        if value.shape != self._shape:
            raise ValueError(f"{self.name} must have shape {self._shape}")
        if not np.isclose(np.linalg.norm(value), 1.0):
            raise ValueError(f"{self.name} must be a unit quaternion")
        setattr(obj, self.private_name, value)


class PoseMessage:
    """
    Class for Pose message data.
    """

    TYPE = Pose
    quaternion = QuaternionDescriptor()

    def __init__(
        self,
    ):
        """
        Initialize the PoseMessage object.
        """
        self.frame_number: int = 0
        self.__pose = np.zeros((4, 4))
        self.__pose[3, 3] = 1.0
        self.quaternion = np.array([1.0, 0.0, 0.0, 0.0])

    @property
    def position(self) -> np.ndarray:
        """
        Return the position vector.
        """
        return self.__pose[:3, 3]

    @property
    def directors(self) -> np.ndarray:
        """
        Return the directors matrix.
        """
        return self.__pose[:3, :3]

    def from_vicon(self, msg: Pose) -> bool:
        """
        Read the Pose message data.
        """
        if (
            msg.x_rot == 0.0
            and msg.y_rot == 0.0
            and msg.z_rot == 0.0
            and msg.w == 0.0
        ):
            return False
        self.frame_number = msg.frame_number
        self.quaternion = np.array([msg.x_rot, msg.y_rot, msg.z_rot, msg.w])
        self.__pose[:3, 3] = np.array([msg.x_trans, msg.y_trans, msg.z_trans])
        self.__pose[:3, :3] = Rotation.from_quat(self.quaternion).as_matrix()
        return True

    def __str__(self) -> str:
        """
        Return the string information of the PoseMsg object.
        """
        return (
            f"\nPoseMessage(\n"
            f"    frame_number={self.frame_number},\n"
            f"    quaternion={np.array2string(self.quaternion, precision=4, suppress_small=True)},\n"
            f"    pose=\n{np.array2string(self.__pose, precision=4, suppress_small=True)},\n"
            f")"
        )
