from typing import Tuple, Union

from dataclasses import dataclass

import numpy as np

from ros2_vicon.array import NDArrayMessage

try:
    import rclpy
    from rclpy.node import Node
except ModuleNotFoundError:
    print(
        "Could not import ROS2 modules. Make sure to source ROS2 workspace first."
    )
    import sys

    sys.exit(1)


@dataclass
class NDArrayPublisher:
    topic: str
    shape: Tuple[int]
    axis_labels: Tuple[str]
    qos_profile: Union[rclpy.qos.QoSProfile, int]
    node: Node

    def __post_init__(self):
        self.__message = NDArrayMessage(
            shape=self.shape,
            axis_labels=self.axis_labels,
        )
        self.__publishing = self.node.create_publisher(
            msg_type=self.__message.TYPE,
            topic=self.topic,
            qos_profile=self.qos_profile,
        )

    def release(self, data: np.ndarray) -> None:
        self.__publishing.publish(self.__message.from_numpy_ndarray(data))

    def __str__(self) -> str:
        """
        Return the string information of the NDArrayPublisher
        """
        return (
            f"NDArrayPublisher(topic={self.topic}, "
            f"message={self.__message}, "
            f"publishing={self.__publishing})"
        )


@dataclass
class PosePublisher:
    topic: str
    qos_profile: Union[rclpy.qos.QoSProfile, int]
    node: Node
    length: int = 1

    def __post_init__(self):
        self.__message = NDArrayMessage(
            shape=(4, 4, self.length),
            axis_labels=["pose", "", "element"],
        )
        self.__publishing = self.node.create_publisher(
            msg_type=self.__message.TYPE,
            topic=self.topic,
            qos_profile=self.qos_profile,
        )

    def release(self, data: np.ndarray) -> None:
        self.__publishing.publish(self.__message.from_numpy_ndarray(data))

    def __str__(self) -> str:
        """
        Return the string information of the PosesPublisher
        """
        return (
            f"PosesPublisher(topic={self.topic}, "
            f"message={self.__message}, "
            f"publishing={self.__publishing})"
        )
