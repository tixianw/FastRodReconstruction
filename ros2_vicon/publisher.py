import numpy as np
from typing import Tuple
from dataclasses import dataclass, field

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Float32MultiArray, MultiArrayDimension
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
            f"    data={np.array2string(self.__data, precision=4, suppress_small=True)}\n"  
            f")"
        )

@dataclass
class NDArrayPublisher:
    topic: str
    message: NDArrayMessage
    publishing: rclpy.publisher.Publisher

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
