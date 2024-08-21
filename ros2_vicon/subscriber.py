import numpy as np
from dataclasses import dataclass

try:
    import rclpy
    from rclpy.node import Node
except ModuleNotFoundError:
    print('Could not import ROS2 modules. Make sure to source ROS2 workspace first.')
    import sys
    sys.exit(1)

class NDArrayDescriptor:
    def __init__(self, shape):
        self.shape = shape

    def __set_name__(self, owner, name):
        self.name = "__" + name

    def __get__(self, obj, objtype=None) -> np.ndarray:
        if obj is None:
            return self
        return obj.__dict__[self.name]

    def __set__(self, obj, value):
        if not isinstance(value, np.ndarray):
            if isinstance(value, list):
                value = np.array(value)
            else:
                raise TypeError(f"{self.name} must be a numpy array")
        if value.shape != self.shape:
            raise ValueError(f"{self.name} must have shape {self.shape}")
        obj.__dict__[self.name] = value

class PoseMsg:
    position = NDArrayDescriptor((3,))
    quaternion = NDArrayDescriptor((4,))

    def __init__(self,):
        # Set initial values
        self.frame_number: int = 0
        self.position = np.zeros(3)
        self.quaternion = np.zeros(4)

    @property
    def directors(self):
        return np.array([self.position, self.position, self.position])
    
    def __repr__(self):
        return (f"\nPositionMsg(\n"
                f"    frame_number={self.frame_number},\n"
                f"    position={np.array2string(self.position, precision=4, suppress_small=True)},\n"
                f"    quaternion={np.array2string(self.quaternion, precision=4, suppress_small=True)},\n"
                f")"
        )

@dataclass
class PoseSubscriber:
    topic: str
    data: PoseMsg
    subscription: rclpy.subscription.Subscription