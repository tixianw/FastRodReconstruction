import numpy as np
from typing import Tuple, Union
from dataclasses import dataclass

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Float32MultiArray, MultiArrayDimension  # For publishing numpy array
except ModuleNotFoundError:
    print('Could not import ROS2 modules. Make sure to source ROS2 workspace first.')
    import sys
    sys.exit(1)

class NDArrayMessage:
    TYPE = Float32MultiArray

    def __init__(self, shape: Tuple[int], axis_labels: Tuple[str]):
        self.shape = shape
        self.axis_labels = axis_labels

        self.__message = self.TYPE()
        self.__message.layout.dim = []
        for i, label in enumerate(axis_labels):
            stride = 1
            for j in range(i, len(shape)):
                stride *= shape[j]
            dim = MultiArrayDimension(
                label=label,
                size=shape[i],
                stride=stride,
            )
            self.__message.layout.dim.append(dim)
        self.__message.layout.data_offset = 0
        self.__data: np.ndarray = np.empty(self.shape)

    def __call__(self, data: np.ndarray) -> Float32MultiArray:
        self.__data = data
        self.__message.data = data.flatten().tolist()
        return self.__message
    
    def __repr__(self):
        return (
            f"\nNDArrayMessage(\n"
            f"    shape={self.shape},\n"
            f"    axis_labels={self.axis_labels},\n"
            f"    data=\n{np.array2string(self.__data, precision=4, suppress_small=True)}\n"  
            f")"
        )

@dataclass
class NDArrayPublisher:
    topic: str
    shape: Tuple[int]
    axis_labels: Tuple[str]
    qos_profile: Union[rclpy.qos.QoSProfile, int]
    node: Node

    def __post_init__(self):
        self.message = NDArrayMessage(
            shape=self.shape,
            axis_labels=self.axis_labels,
        )
        self.publishing = self.node.create_publisher(
            msg_type=NDArrayMessage.TYPE,
            topic=self.topic,
            qos_profile=self.qos_profile,
        )

    def publish(self, data: np.ndarray):
        self.publishing.publish(
            self.message(data)
        )

    def __repr__(self):
        return (
            f"NDArrayPublisher(topic={self.topic}, "
            f"message={self.message}, "
            f"publishing={self.publishing})"
        )
