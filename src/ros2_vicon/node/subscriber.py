from typing import Callable, Union

from dataclasses import dataclass

from ros2_vicon.message.pose import PoseMessage

try:
    import rclpy
    from rclpy.node import Node
    from vicon_receiver.msg import Position as Pose
except ModuleNotFoundError:
    print(
        "Could not import ROS2 modules. Make sure to source ROS2 workspace first."
    )
    import sys

    sys.exit(1)


@dataclass
class PoseSubscriber:
    """
    Dataclass for Pose message subscriber.
    """

    topic: str
    callback: Callable[[Pose], None]
    qos_profile: Union[rclpy.qos.QoSProfile, int]
    node: Node

    def __post_init__(self):
        """
        Initialize the PoseSubscriber object.
        """
        self.__message = PoseMessage()
        self.__subscription = self.node.create_subscription(
            msg_type=self.__message.TYPE,
            topic=self.topic,
            callback=self.callback,
            qos_profile=self.qos_profile,
        )

    def receive(self, msg: Pose) -> bool:
        """
        Read the Pose message data.
        """
        return self.__message.from_vicon(msg)

    @property
    def message(self) -> PoseMessage:
        """
        Return the Pose message data.
        """
        return self.__message

    def __str__(self) -> str:
        """
        Return the string information of the PoseSubscriber
        """
        return (
            f"PoseSubscriber(topic={self.topic}, "
            f"message={self.__message}, "
            f"subscription={self.__subscription})"
        )
