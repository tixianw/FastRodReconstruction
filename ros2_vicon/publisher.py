from typing import Tuple
from dataclasses import dataclass, field

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Float32MultiArray
except ModuleNotFoundError:
    print('Could not import ROS2 modules. Make sure to source ROS2 workspace first.')
    import sys
    sys.exit(1)

class NDArrayMessage:
    TYPE = Float32MultiArray

    def __init__(self, topic: str, shape: Tuple[int], axis_labels: Tuple[str]):
        self.topic = topic
        self.shape = shape
        self.axis_labels = axis_labels