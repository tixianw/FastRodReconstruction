from typing import Dict, List, Optional, Tuple, Union

from dataclasses import dataclass

import click
import numpy as np

from neural_data_smoothing3D import pos_dir_to_input
from reconstruction import ReconstructionModel, ReconstructionResult
from ros2_vicon import NDArrayPublisher, PosePublisher, PoseSubscriber, Timer

try:
    import rclpy
    from rclpy.node import Node
except ModuleNotFoundError:
    print(
        "Could not import ROS2 modules. Make sure to source ROS2 workspace first."
    )
    import sys

    sys.exit(1)


@dataclass
class ReconstructionModelResource:
    data_file_name: Optional[str] = None
    model_file_name: Optional[str] = None


class ReconstructionNode(Node):
    def __init__(
        self,
        subscription_topics: Tuple[str],
        reconstruction_rate: float = 60.0,
        model_resource: Optional[ReconstructionModelResource] = None,
        log_level: str = "info",
    ):
        super().__init__("reconstruction_node")
        self.set_logger_level(log_level)
        self.get_logger().info("Reconstruction node initializing...")

        self.subscription_topics = subscription_topics
        self.reconstruction_rate = reconstruction_rate

        # Initialize subscribers
        self.get_logger().info("- Subcribers initializing...")
        self.__subscribers: List[PoseSubscriber] = []
        for i, topic in enumerate(self.subscription_topics):
            subscriber = PoseSubscriber(
                topic=topic,
                callback=self.subscriber_callback_closure(i),
                qos_profile=100,
                node=self,
            )
            self.__subscribers.append(subscriber)
        self.init_input_data()

        # Initialize reconstruction model
        self.get_logger().info("- Reconstruction model initializing...")
        self.model = (
            ReconstructionModel(
                data_file_name=model_resource.data_file_name,
                model_file_name=model_resource.model_file_name,
            )
            if model_resource
            else ReconstructionModel()
        )

        # Initialize publishers
        self.get_logger().info("- Publishers initializing...")
        self.__publishers: Dict[str, Union[PosePublisher, NDArrayPublisher]] = {
            "pose": PosePublisher(
                topic="/vicon/pose",
                length=self.number_of_markers - 1,
                qos_profile=100,
                node=self,
            ),
            "position": NDArrayPublisher(
                topic="/reconstruction/position",
                shape=self.model.result.position.shape,
                axis_labels=("position", "element"),
                qos_profile=100,
                node=self,
            ),
            "directors": NDArrayPublisher(
                topic="/reconstruction/directors",
                shape=self.model.result.directors.shape,
                axis_labels=("directors", "director_index", "element"),
                qos_profile=100,
                node=self,
            ),
            "kappa": NDArrayPublisher(
                topic="/reconstruction/kappa",
                shape=self.model.result.kappa.shape,
                axis_labels=("kappa", "element"),
                qos_profile=100,
                node=self,
            ),
        }

        # Create a timer for publishing at reconstruction_rate Hz
        # self.timer = self.create_timer(
        #     timer_period_sec=1 / self.reconstruction_rate,
        #     callback=self.timer_callback,
        # )
        self.timer = Timer(
            timer_period_sec=1 / self.reconstruction_rate,
            callback=self.timer_callback,
            node=self,
            publish_flag=True,
            qos_profile=100,
        )

    def init_input_data(self) -> None:
        self.number_of_markers = len(self.__subscribers)
        self.input_data = np.zeros((1, 4, 4, self.number_of_markers - 1))
        self.input_data[0, 3, 3, :] = 1.0
        self.input_data[0, :3, :3, :] = np.eye(3)

    def set_logger_level(self, log_level: str) -> None:
        level_map = {
            "debug": rclpy.logging.LoggingSeverity.DEBUG,
            "info": rclpy.logging.LoggingSeverity.INFO,
            "warn": rclpy.logging.LoggingSeverity.WARN,
            "error": rclpy.logging.LoggingSeverity.ERROR,
            "fatal": rclpy.logging.LoggingSeverity.FATAL,
        }
        assert (
            log_level.lower() in level_map.keys()
        ), f"Invalid log level: {log_level}"
        level = level_map[log_level.lower()]
        self.get_logger().set_level(level)

    @property
    def result(self) -> ReconstructionResult:
        return self.model.result

    def subscriber_callback_closure(self, i: int) -> callable:
        def subscriber_callback(msg):
            self.__subscribers[i].receive(msg)
            self.get_logger().debug(f"{self.__subscribers[i]}")

            # self.get_logger().debug(f'{msg.frame_number}')
            # self.get_logger().debug(f'  {msg.x_trans}')
            # self.get_logger().debug(f'  {msg.y_trans}')
            # self.get_logger().debug(f'  {msg.z_trans}')
            # self.get_logger().debug(f'  {msg.x_rot}')
            # self.get_logger().debug(f'  {msg.y_rot}')
            # self.get_logger().debug(f'  {msg.z_rot}')
            # self.get_logger().debug(f'  {msg.w}')

        return subscriber_callback

    def timer_callback(self) -> None:
        self.reconstruct()
        self.publish("pose", self.input_data[0])
        self.publish("position", self.result.position)
        self.publish("directors", self.result.directors)
        self.publish("kappa", self.result.kappa)

    def create_input_data(self) -> np.ndarray:
        # Create input data from the subscribers
        base_position = self.__subscribers[0].message.position
        base_directors = self.__subscribers[0].message.directors
        for i, subscriber in enumerate(self.__subscribers[1:]):
            self.input_data[0, :3, 3, i] = (
                subscriber.message.position - base_position
            )
            self.input_data[0, :3, :3, i] = subscriber.message.directors

        input_data = pos_dir_to_input(
            pos=self.input_data[:, :3, 3, :],
            dir=self.input_data[:, :3, :3, :],
        )
        return input_data

    def reconstruct(self):
        self.model(self.create_input_data())

    def publish(self, publisher_key: str, data: np.ndarray) -> None:
        self.__publishers[publisher_key].release(data)
        self.get_logger().debug(f"{self.__publishers[publisher_key]}")


def set_subsciption_topics(source: str):
    if source.lower() == "vicon":
        subscription_topics = (
            "/vicon/br2_seg_1/br2_seg_1",
            "/vicon/br2_seg_2/br2_seg_2",
            "/vicon/br2_seg_3/br2_seg_3",
            "/vicon/br2_seg_4/br2_seg_4",
        )
    if source.lower() == "vicon_mock":
        subscription_topics = (
            "/vicon_mock/CrossSection_0_0/CrossSection_0_0",
            "/vicon_mock/CrossSection_0_1/CrossSection_0_1",
            "/vicon_mock/CrossSection_0_2/CrossSection_0_2",
            "/vicon_mock/CrossSection_0_3/CrossSection_0_3",
            # "/vicon_mock/CrossSection_0_4/CrossSection_0_4",
            # "/vicon_mock/CrossSection_0_5/CrossSection_0_5",
        )
    return subscription_topics


@click.command()
@click.option(
    "--log-level",
    type=click.Choice(
        ["debug", "info", "warn", "error", "fatal"],
        case_sensitive=False,
    ),
    default="info",
    help="Set the logging level",
)
@click.option(
    "--source",
    type=click.Choice(
        ["vicon", "vicon_mock"],
        case_sensitive=False,
    ),
    default="vicon",
    help="Set the source of the listener",
)
def main(log_level: str, source: str):
    rclpy.init()
    node = ReconstructionNode(
        subscription_topics=set_subsciption_topics(source),
        log_level=log_level,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        # Destroy the node explicitly
        # (optional - otherwise it will be done automatically
        # when the garbage collector destroys the node object)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
