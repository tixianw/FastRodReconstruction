from typing import Tuple

import numpy as np

try:
    from std_msgs.msg import (  # For publishing numpy array
        Float32MultiArray,
        MultiArrayDimension,
    )
except ModuleNotFoundError:
    print(
        "Could not import ROS2 modules. Make sure to source ROS2 workspace first."
    )
    import sys

    sys.exit(1)


class NDArrayDescriptor:
    """
    Descriptor for numpy arrays.
    """

    def __init__(self, shape=None):
        self._shape = shape

    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name
        self.private_name = "__" + name

    def __get__(self, obj: object, objtype: type) -> np.ndarray:
        value: np.ndarray = getattr(obj, self.private_name)
        return value

    def __set__(self, obj: object, value: np.ndarray) -> None:
        assert isinstance(
            value, np.ndarray
        ), f"{self.name} must be a numpy array"
        if self._shape and value.shape != self._shape:
            raise ValueError(f"{self.name} must have shape {self._shape}")
        setattr(obj, self.private_name, value)


class NDArrayMessage:
    TYPE = Float32MultiArray
    data = NDArrayDescriptor()

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
        self.data = np.empty(self.shape)

    def from_topic(self, msg: Float32MultiArray) -> bool:
        try:
            self.from_message(msg)
        except AssertionError:
            return False
        return True

    def from_numpy_ndarray(self, data: np.ndarray) -> "NDArrayMessage":
        assert (
            data.shape == self.shape
        ), f"Data shape {data.shape} must be {self.shape}"
        self.data = data
        self.__message.data = data.flatten().tolist()
        return self

    def to_message(self) -> Float32MultiArray:
        return self.__message

    def from_message(self, msg: Float32MultiArray) -> "NDArrayMessage":
        shape = tuple(dim.size for dim in msg.layout.dim)
        assert (
            shape == self.shape
        ), f"Shape of msg: {shape} does not align with {self.shape}"
        axis_labels = tuple(dim.label for dim in msg.layout.dim)
        assert (
            axis_labels == self.axis_labels
        ), f"Axis labels of msg: {axis_labels} does not align with {self.axis_labels}"
        self.__message = msg
        self.data = np.array(msg.data).reshape(self.shape)
        return self

    def to_numpy_ndarray(self) -> np.ndarray:
        return self.data

    def to_hdf5(self) -> np.ndarray:
        return self.data.copy()

    def add_info(self, key: str, value: str) -> "NDArrayMessage":
        setattr(self.__message, key, value)
        return self

    def __str__(self) -> str:
        """
        Return the string information of the NDArrayMessage
        """
        return (
            f"\nNDArrayMessage(\n"
            f"    shape={self.shape},\n"
            f"    axis_labels={self.axis_labels},\n"
            f"    data=\n{np.array2string(self.data, precision=4, suppress_small=True)}\n"
            f")"
        )
