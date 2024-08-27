from typing import Union

import numpy as np
from scipy.spatial.transform import Rotation

from ros2_vicon.message.pose import PoseDescriptor, QuaternionDescriptor

try:
    from vicon_receiver.msg import Position as ViconPose
except ModuleNotFoundError:
    print(
        "Could not import ROS2 modules. Make sure to source ROS2 workspace first."
    )
    import sys

    sys.exit(1)


class ViconPoseMessage:
    """
    Class for Vicon pose message data.
    """

    TYPE = ViconPose
    quaternion = QuaternionDescriptor()
    pose = PoseDescriptor()

    def __init__(self):
        """
        Initialize the ViconPoseMessage object.
        """
        self.frame_number: int = 0
        self.quaternion = np.array([1.0, 0.0, 0.0, 0.0])
        self.pose = np.eye(4)

    @property
    def position(self) -> np.ndarray:
        """
        Return the position vector.
        """
        return self.pose[:3, 3]

    @property
    def directors(self) -> np.ndarray:
        """
        Return the directors matrix.
        """
        return self.pose[:3, :3]

    def from_topic(self, msg: ViconPose) -> bool:
        """
        Read the Vicon pose message data.
        """
        try:
            self.quaternion = np.array([msg.x_rot, msg.y_rot, msg.z_rot, msg.w])
        except ValueError:
            return False
        self.pose[:3, 3] = (
            np.array([msg.x_trans, msg.y_trans, msg.z_trans]) / 1000
        )  # Convert from millimeters to meters
        self.pose[:3, :3] = Rotation.from_quat(
            self.quaternion
        ).as_matrix()  # Convert from quaternion to rotation matrix
        self.frame_number: int = msg.frame_number
        return True

    def __str__(self) -> str:
        """
        Return the string information of the PoseMsg object.
        """
        return (
            f"\nViconPoseMessage(\n"
            f"    frame_number={self.frame_number},\n"
            f"    quaternion={np.array2string(self.quaternion, precision=4, suppress_small=True)},\n"
            f"    pose=\n{np.array2string(self.pose, precision=4, suppress_small=True)},\n"
            f")"
        )
