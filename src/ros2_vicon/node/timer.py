from typing import ClassVar, Union

from dataclasses import dataclass
from enum import Enum

import numpy as np

from ros2_vicon.message.array import NDArrayMessage
from ros2_vicon.node import LoggerNode
from ros2_vicon.qos import QoSProfile


@dataclass
class Timer:
    timer_period_sec: float
    callback: callable
    node: LoggerNode
    topic: str = "/time"
    publish_flag: bool = False
    qos_profile: Union[QoSProfile, int] = 1

    class PUBLISH_TIME(Enum):
        TRUE = True
        FALSE = False

    def __post_init__(self):

        self.__init_time = self.get_clock()
        self.__time = NDArrayMessage(
            shape=(1,),
            axis_labels=("time",),
        )
        if self.publish_flag:
            self.__publisher = self.node.create_publisher(
                msg_type=self.__time.TYPE,
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
        return self.__time.from_numpy_ndarray(
            data=np.array([self.get_clock() - self.__init_time])
        ).to_numpy_ndarray()

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

    def __callback_with_publish(self) -> None:
        """
        Callback function with publishing the current time.
        """
        if self.callback():
            self.__publisher.publish(
                self.__time.from_numpy_ndarray(data=self.time).to_message()
            )
            self.node.log_debug(f"{self}")

    def __callback_without_publish(self) -> bool:
        """
        Callback function without publishing the current time.
        """
        self.callback()

    def get_clock(self) -> float:
        """
        Get the current time.
        """
        return self.node.get_clock().now().nanoseconds / 1e9
