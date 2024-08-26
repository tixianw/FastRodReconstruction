from typing import Union

from dataclasses import dataclass

from ros2_vicon.node import LoggerNode
from ros2_vicon.qos import QoSProfile

try:
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
    node: LoggerNode
    topic: str = "/time"
    publish_flag: bool = False
    qos_profile: Union[QoSProfile, int] = 1

    def __post_init__(self):

        self.__init_time = self.get_clock()
        self.__time = 0.0
        if self.publish_flag:
            self.__publisher = self.node.create_publisher(
                msg_type=Float32,
                topic=self.topic,
                qos_profile=self.qos_profile,
            )
            callback = self.__callback_with_publish
        else:
            callback = self.__callback_without_publish

        self.__timer = self.node.create_timer(
            timer_period_sec=self.timer_period_sec,
            callback=callback,
        )

    @property
    def time(self) -> float:
        """
        Get the current time.
        """
        self.__time = self.get_clock() - self.__init_time
        return self.__time

    def __str__(self) -> str:
        """
        Return the string information of the Timer
        """

        if self.publish_flag:
            info = (
                f"Timer(timer_period_sec={self.timer_period_sec}, \n"
                f"timer={self.__timer}, \n"
                f"publisher={self.__publisher}, \n"
                f"topic={self.topic}, \n"
                f"time={self.time})"
            )
        else:
            info = (
                f"Timer(timer_period_sec={self.timer_period_sec}, "
                f"timer={self.time})"
            )
        return info

    def __callback_with_publish(self) -> bool:
        """
        Callback function with publishing the current time.
        """
        if self.callback():
            self.__publisher.publish(Float32(data=self.time))
            self.node.log_debug(f"{self}")
            return True
        return False

    def __callback_without_publish(self) -> bool:
        """
        Callback function without publishing the current time.
        """
        return self.callback()

    def get_clock(self) -> float:
        """
        Get the current time.
        """
        return self.node.get_clock().now().nanoseconds / 1e9
