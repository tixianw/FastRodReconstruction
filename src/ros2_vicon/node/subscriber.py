from typing import Callable, Optional, Tuple, Union

from dataclasses import dataclass

from ros2_vicon.message.array import NDArrayMessage
from ros2_vicon.message.pose import PoseMessage
from ros2_vicon.message.vicon import ViconPoseMessage

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Float32MultiArray  # For subscribing numpy array
    from vicon_receiver.msg import Position as ViconPose
except ModuleNotFoundError:
    print(
        "Could not import ROS2 modules. Make sure to source ROS2 workspace first."
    )
    import sys

    sys.exit(1)


@dataclass
class ViconPoseSubscriber:
    """
    Dataclass for Vicon pose message subscriber.
    """

    topic: str
    callback: Callable[[ViconPose], None]
    qos_profile: Union[rclpy.qos.QoSProfile, int]
    node: Optional[Node] = None

    def __post_init__(self):
        """
        Initialize the ViconPoseSubscriber object.
        """
        self.message = ViconPoseMessage()
        self.__subscription = self.node.create_subscription(
            msg_type=self.message.TYPE,
            topic=self.topic,
            callback=self.callback,
            qos_profile=self.qos_profile,
        )

    def receive(self, msg: ViconPose) -> bool:
        """
        Read the Vicon pose message data.
        """
        return self.message.from_topic(msg)

    def __str__(self) -> str:
        """
        Return the string information of the ViconPoseSubscriber
        """
        return (
            f"ViconPoseSubscriber(topic={self.topic}, "
            f"message={self.message}, "
            f"subscription={self.__subscription})"
        )


@dataclass
class NDArraySubscriber:
    """
    Dataclass for NDArray message subscriber.
    """

    message: NDArrayMessage
    topic: str
    callback: Callable[[Float32MultiArray], None]
    qos_profile: Union[rclpy.qos.QoSProfile, int]
    node: Optional[Node] = None

    def __post_init__(self):
        """
        Initialize the NDArraySubscriber object.
        """
        self.__subscription = self.node.create_subscription(
            msg_type=self.message.TYPE,
            topic=self.topic,
            callback=self.callback,
            qos_profile=self.qos_profile,
        )

    def receive(self, msg: Float32MultiArray) -> bool:
        """
        Read the NDArray message data.
        """
        return self.message.from_topic(msg)

    def __str__(self) -> str:
        """
        Return the string information of the NDArraySubscriber
        """
        return (
            f"NDArraySubscriber(topic={self.topic}, "
            f"message={self.message}, "
            f"subscription={self.__subscription})"
        )


@dataclass
class PoseSubscriber:
    """
    Dataclass for Pose message subscriber.
    """

    message: PoseMessage
    topic: str
    callback: Callable[[Float32MultiArray], None]
    qos_profile: Union[rclpy.qos.QoSProfile, int]
    node: Optional[Node] = None

    def __post_init__(self):
        """
        Initialize the PoseSubscriber object.
        """
        self.__subscription = self.node.create_subscription(
            msg_type=self.message.TYPE,
            topic=self.topic,
            callback=self.callback,
            qos_profile=self.qos_profile,
        )

    def receive(self, msg: Float32MultiArray) -> bool:
        """
        Read the Pose message data.
        """
        return self.message.from_topic(msg)

    def __str__(self) -> str:
        """
        Return the string information of the PoseSubscriber
        """
        return (
            f"PoseSubscriber(topic={self.topic}, "
            f"message={self.message}, "
            f"subscription={self.__subscription})"
        )
