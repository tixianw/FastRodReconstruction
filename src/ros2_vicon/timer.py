from typing import Union

from dataclasses import dataclass

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Float32
except ModuleNotFoundError:
    print(
        "Could not import ROS2 modules. Make sure to source ROS2 workspace first."
    )
    import sys

    sys.exit(1)


@dataclass
class Timer:
    timer_period_sec: float
    callback: callable
    node: Node
    topic: str = "/time"
    publish_flag: bool = False
    qos_profile: Union[rclpy.qos.QoSProfile, int] = 1

    def __post_init__(self):

        if self.publish_flag:
            self.__publisher = self.node.create_publisher(
                msg_type=Float32,
                topic=self.topic,
                qos_profile=self.qos_profile,
            )
            callback = self.__callback_with_publish
            self.__init_time = self.get_clock()
            self.__time = 0.0
        else:
            callback = self.callback

        self.__timer = self.node.create_timer(
            timer_period_sec=self.timer_period_sec,
            callback=callback,
        )

    def __str__(self) -> str:
        """
        Return the string information of the Timer
        """

        if self.publish_flag:
            info = (
                f"Timer(timer_period_sec={self.timer_period_sec}, \n"
                f"timer={self.__timer}, \n"
                f"topic={self.topic}, \n"
                f"time={self.__time})"
            )
        else:
            info = (
                f"Timer(timer_period_sec={self.timer_period_sec}, "
                f"timer={self.__timer})"
            )
        return info

    def __callback_with_publish(self):
        """
        Callback function with publishing the current time.
        """
        self.__time = self.get_clock() - self.__init_time
        self.__publisher.publish(Float32(data=self.__time))
        self.node.get_logger().debug(f"{self}")
        self.callback()

    def get_clock(self) -> float:
        """
        Get the current time.
        """
        return self.node.get_clock().now().nanoseconds / 1e9
