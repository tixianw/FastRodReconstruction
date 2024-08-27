from typing import Dict, List, Optional, Tuple, Union

import time
from dataclasses import dataclass

import click
import numpy as np

import ros2_vicon
from neural_data_smoothing3D import pos_dir_to_input
from reconstruction import ReconstructionModel, ReconstructionResult
from ros2_vicon import (
    NDArrayPublisher,
    PosePublisher,
    Timer,
    ViconPoseSubscriber,
)
from ros2_vicon.node import LoggerNode


@dataclass
class ReconstructionModelResource:
    data_file_name: Optional[str] = None
    model_file_name: Optional[str] = None


class ReconstructionNode(LoggerNode):
    def __init__(
        self,
        subscription_topics: Tuple[str],
        reconstruction_rate: float = 60.0,
        model_resource: Optional[ReconstructionModelResource] = None,
        log_level: str = "info",
    ):
        super().__init__(
            node_name="reconstruction_node",
            log_level=log_level,
        )

        self.subscription_topics = subscription_topics
        self.reconstruction_rate = reconstruction_rate

        # Initialize subscribers
        self.log_info("- Subcribers initializing...")
        self.__subscribers: List[ViconPoseSubscriber] = []
        for i, topic in enumerate(self.subscription_topics):
            subscriber = ViconPoseSubscriber(
                topic=topic,
                callback=self.subscriber_callback_closure(i),
                qos_profile=100,
                node=self,
            )
            self.__subscribers.append(subscriber)
        self.init_input_data()

        # Initialize reconstruction model
        self.log_info("- Reconstruction model initializing...")
        self.model = (
            ReconstructionModel(
                data_file_name=model_resource.data_file_name,
                model_file_name=model_resource.model_file_name,
            )
            if model_resource
            else ReconstructionModel()
        )

        # Initialize publishers
        self.log_info("- Publishers initializing...")
        self.__publishers: Dict[str, Union[PosePublisher, NDArrayPublisher]] = {
            "pose": PosePublisher(
                topic="/vicon/pose",
                length=self.number_of_markers,
                label="element",
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

        # Initialize timer
        self.log_info("- Timer initializing...")
        self.timer = Timer(
            timer_period_sec=1 / self.reconstruction_rate,
            callback=self.timer_callback,
            node=self,
            publish_flag=True,
            qos_profile=100,
        )

        self.ready()
        self.time = time.time()

    def init_input_data(self) -> None:
        self.number_of_markers = len(self.__subscribers)
        self.input_data = np.zeros((1, 4, 4, self.number_of_markers))
        self.input_data[0, 3, 3, :] = 1.0
        for i in range(self.number_of_markers):
            self.input_data[0, :3, :3, i] = np.diag([1, -1, -1])
        self.new_message = False

    @property
    def result(self) -> ReconstructionResult:
        return self.model.result

    def subscriber_callback_closure(self, i: int) -> callable:
        def subscriber_callback(msg):
            self.new_message = self.__subscribers[i].receive(msg)
            self.log_debug(f"{self.__subscribers[i]}")

            # self.log_debug(f'{msg.frame_number}')
            # self.log_debug(f'{msg.x_trans}')
            # self.log_debug(f'{msg.y_trans}')
            # self.log_debug(f'{msg.z_trans}')
            # self.log_debug(f'{msg.x_rot}')
            # self.log_debug(f'{msg.y_rot}')
            # self.log_debug(f'{msg.z_rot}')
            # self.log_debug(f'{msg.w}')

        return subscriber_callback

    def timer_callback(self) -> bool:
        if not self.new_message:
            return False
        new_time = time.time()
        self.log_debug(f"Reconstructing...")
        self.reconstruct()
        self.log_info(
            f"Reconstructing... at rate: {1/(new_time-self.time):.2f} Hz. Done!"
        )
        self.time = new_time
        self.publish("pose", self.input_data[0])
        self.publish("position", self.result.position)
        self.publish("directors", self.result.directors)
        self.publish("kappa", self.result.kappa)
        self.new_message = False
        return True

    def create_input_data(self) -> np.ndarray:
        # Create input data from the subscribers
        self.model.base_position = self.__subscribers[0].message.position
        self.model.base_directors = self.__subscribers[0].message.directors
        for i, subscriber in enumerate(self.__subscribers):
            self.input_data[0, :3, 3, i] = subscriber.message.position.copy()
            self.input_data[0, :3, :3, i] = subscriber.message.directors.copy()

        input_data = pos_dir_to_input(
            pos=self.model.remove_base_translation(
                marker_position=self.input_data[:, :3, 3, 1:]
            ),
            dir=self.model.remove_base_rotation(
                marker_directors=self.input_data[:, :3, :3, 1:]
            ),
        )
        return input_data

    def reconstruct(self):
        self.model(self.create_input_data())

    def publish(
        self,
        publisher_key: str,
        data: np.ndarray,
    ) -> None:
        self.__publishers[publisher_key].release(data)
        self.log_debug(f"{self.__publishers[publisher_key]}")


def set_subsciption_topics(source: str):
    if source.lower() == "vicon":
        subscription_topics = (
            "/vicon/br2_seg_1/br2_seg_1",
            "/vicon/br2_seg_2/br2_seg_2",
            "/vicon/br2_seg_3/br2_seg_3",
            # "/vicon/br2_seg_4/br2_seg_4",
        )
    if source.lower() == "vicon_mock":
        subscription_topics = (
            "/vicon_mock/CrossSection_0_0/CrossSection_0_0",
            "/vicon_mock/CrossSection_0_1/CrossSection_0_1",
            "/vicon_mock/CrossSection_0_2/CrossSection_0_2",
            # "/vicon_mock/CrossSection_0_3/CrossSection_0_3",
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
    help="Set the source of the subscriber",
)
def main(log_level: str, source: str):
    ros2_vicon.init()

    node = ReconstructionNode(
        subscription_topics=set_subsciption_topics(source),
        log_level=log_level,
    )
    try:
        node.start()
    except KeyboardInterrupt:
        node.stop()
        ros2_vicon.shutdown()


if __name__ == "__main__":
    main()
