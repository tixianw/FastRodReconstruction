from typing import Tuple, Union

from dataclasses import dataclass

import numpy as np

from ros2_vicon.message.array import NDArrayMessage
from ros2_vicon.message.pose import PoseMessage
from ros2_vicon.node import LoggerNode
from ros2_vicon.qos import QoSProfile


@dataclass
class NDArrayPublisher:
    topic: str
    shape: Tuple[int]
    axis_labels: Tuple[str]
    qos_profile: Union[QoSProfile, int]
    node: LoggerNode

    def __post_init__(self):
        self.__message = NDArrayMessage(
            shape=self.shape,
            axis_labels=self.axis_labels,
        )
        self.__publisher = self.node.create_publisher(
            msg_type=self.__message.TYPE,
            topic=self.topic,
            qos_profile=self.qos_profile,
        )

    def release(self, data: np.ndarray) -> None:
        self.__publisher.publish(
            self.__message.from_numpy_ndarray(data).to_message()
        )

    def __str__(self) -> str:
        """
        Return the string information of the NDArrayPublisher
        """
        return (
            f"NDArrayPublisher(topic={self.topic}, "
            f"message={self.__message}, "
            f"publisher={self.__publisher})"
        )


@dataclass
class PosePublisher:
    topic: str
    qos_profile: Union[QoSProfile, int]
    node: LoggerNode
    length: int = 1
    label: str = "element"

    def __post_init__(self):
        self.__message = PoseMessage(
            shape=(self.length,),
            axis_labels=(self.label,),
        )
        self.__publisher = self.node.create_publisher(
            msg_type=self.__message.TYPE,
            topic=self.topic,
            qos_profile=self.qos_profile,
        )

    def release(self, data: np.ndarray) -> None:
        self.__publisher.publish(
            self.__message.from_numpy_ndarray(data).to_message()
        )

    def __str__(self) -> str:
        """
        Return the string information of the PosesPublisher
        """
        return (
            f"PosesPublisher(topic={self.topic}, "
            f"message={self.__message}, "
            f"publisher={self.__publisher})"
        )
