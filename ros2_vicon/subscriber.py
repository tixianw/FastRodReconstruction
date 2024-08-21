import numpy as np
from dataclasses import dataclass
from typing import Union
from scipy.spatial.transform import Rotation

try:
    import rclpy
    from rclpy.node import Node
    from vicon_receiver.msg import Position as Pose
except ModuleNotFoundError:
    print('Could not import ROS2 modules. Make sure to source ROS2 workspace first.')
    import sys
    sys.exit(1)


class NDArrayDescriptor:
    """
    Descriptor for numpy arrays.
    """

    def __init__(self, shape):
        self.shape = shape

    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name
        self.private_name = "__" + name

    def __get__(self, obj: object, objtype: type) -> np.ndarray:
        value: np.ndarray = getattr(obj, self.private_name)
        return value

    def __set__(self, obj: object, value: Union[np.ndarray, list]) -> None:
        if isinstance(value, list):
            value = np.array(value)
        if not isinstance(value, np.ndarray):
            raise TypeError(f"{self.name} must be a numpy array or list")
        if value.shape != self.shape:
            raise ValueError(f"{self.name} must have shape {self.shape}")
        setattr(obj, self.private_name, value)


class PoseMessage:
    """
    Class for Pose message data.
    """
    TYPE = Pose
    position = NDArrayDescriptor((3,))
    quaternion = NDArrayDescriptor((4,))

    def __init__(self,):
        """
        Initialize the PoseMessage object.
        """ 
        self.frame_number: int = 0
        self.position = np.zeros(3)
        self.quaternion = np.array([1.0, 0.0, 0.0, 0.0])

    @property
    def directors(self) -> np.ndarray:
        """
        Convert quaternion to directors matrix (rotation matrix).
        Returns a 3x3 numpy array representing the rotation.
        """
        return Rotation.from_quat(self.quaternion).as_matrix()
    
    def __repr__(self) -> str:
        """
        Return the string representation of the PoseMsg object.
        """
        return (
            f"\nPoseMessage(\n"
            f"    frame_number={self.frame_number},\n"
            f"    position={np.array2string(self.position, precision=4, suppress_small=True)},\n"
            f"    quaternion={np.array2string(self.quaternion, precision=4, suppress_small=True)},\n"
            f")"
        )


@dataclass
class PoseSubscriber:
    """
    Dataclass for Pose message subscriber.
    """

    topic: str
    data: PoseMessage
    subscription: rclpy.subscription.Subscription
