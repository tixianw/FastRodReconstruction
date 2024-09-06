from typing import Dict, List, Optional, Tuple, Union

import time
from collections import OrderedDict
from dataclasses import dataclass

import click
import numpy as np

import ros2_vicon
from reconstruction import ReconstructionModel
from ros2_vicon import (
    NDArrayPublisher,
    PosePublisher,
    Timer,
    ViconPoseSubscriber,
)
from ros2_vicon.filter import PoseFilter
from ros2_vicon.node import StageNode


@dataclass
class ReconstructionModelResource:
    data_file_name: Optional[str] = None
    model_file_name: Optional[str] = None


class ReconstructionNode(StageNode):
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
            ordered_stages=OrderedDict(
                filter_transition=self.stage_filter_transition,
                lab_frame_calibration=self.stage_lab_frame_calibration,
                material_frame_calibration=self.stage_material_frame_calibration,
                reconstruction=self.stage_reconstruction,
            ),
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
        self.number_of_markers = len(self.__subscribers)
        self.pose = np.repeat(
            np.expand_dims(
                np.diag([1.0, -1.0, -1.0, 1.0]),
                axis=2,
            ),
            self.number_of_markers,
            axis=2,
        )
        self.new_message = False

        # Initialize pose filter
        self.log_info("- Pose filter initializing...")
        self.filter = PoseFilter(
            director_filter_gain=0.1 * np.ones(self.number_of_markers),
            position_filter_gain=0.1 * np.ones(self.number_of_markers),
        )

        # Initialize reconstruction model
        self.log_info("- Reconstruction model initializing...")
        self.model = (
            ReconstructionModel(
                data_file_name=model_resource.data_file_name,
                model_file_name=model_resource.model_file_name,
            )
            if model_resource
            else ReconstructionModel(number_of_markers=self.number_of_markers)
        )
        self.log_info(f"\n{self.model}")

        # Initialize publishers
        self.log_info("- Publishers initializing...")
        self.__publishers: Dict[str, Union[PosePublisher, NDArrayPublisher]] = {
            "pose": PosePublisher(
                topic="/vicon/pose",
                length=self.number_of_markers,
                label="marker",
                qos_profile=100,
                node=self,
            ),
            "filtered_pose": PosePublisher(
                topic="/filter/pose",
                length=self.number_of_markers,
                label="marker",
                qos_profile=100,
                node=self,
            ),
            "lab_frame_transformation": NDArrayPublisher(
                topic="/reconstruction/lab_frame_transformation",
                shape=(4, 4),
                axis_labels=("transformation_matrix", ""),
                qos_profile=100,
                node=self,
            ),
            "material_frame_transformation": NDArrayPublisher(
                topic="/reconstruction/material_frame_transformation",
                shape=(4, 4, self.number_of_markers),
                axis_labels=("transformation_matrix", "", "marker"),
                qos_profile=100,
                node=self,
            ),
            "calibrated_pose": PosePublisher(
                topic="/reconstruction/calibrated_pose",
                length=self.number_of_markers,
                label="marker",
                qos_profile=100,
                node=self,
            ),
            "input": NDArrayPublisher(
                topic="/reconstruction/input",
                shape=self.model.input_pose.shape,
                axis_labels=("input", "", "marker"),
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

    def create_pose(self) -> None:
        # Create pose from the subscribers
        for i, subscriber in enumerate(self.__subscribers):
            self.pose[:3, 3, i] = subscriber.message.position.copy()
            self.pose[:3, :3, i] = subscriber.message.directors.copy()

    def timer_callback(self) -> Timer.PUBLISH_TIME:
        if not self.new_message:
            return False
        self.create_pose()
        self.filter.update(self.pose)
        self.new_message = False
        return self.stage()

    def stage_filter_transition(self) -> Timer.PUBLISH_TIME:
        if time.time() - self.time > 2:
            self.next_stage()
            self.time = time.time()
        return self.timer.PUBLISH_TIME.FALSE

    def stage_lab_frame_calibration(self) -> Timer.PUBLISH_TIME:
        lab_angle = -100.0 / 180.0 * np.pi  # 155
        self.model.lab_frame_transformation[:3, :3] = np.array(
            [
                [np.cos(lab_angle), -np.sin(lab_angle), 0.0],
                [np.sin(lab_angle), np.cos(lab_angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        material_angle = lab_angle
        for i in range(self.number_of_markers):
            self.model.material_frame_transformation[:3, :3, i] = np.array(
                [
                    [np.cos(material_angle), -np.sin(material_angle), 0.0],
                    [np.sin(material_angle), np.cos(material_angle), 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )

        if self.model.process_lab_frame_calibration(self.filter.pose):
            self.next_stage()
        return self.timer.PUBLISH_TIME.FALSE

    def stage_material_frame_calibration(self) -> Timer.PUBLISH_TIME:
        # for i in range(self.number_of_markers):
        #     self.model.material_frame_transformation[:2, 3, i] = np.array(
        #         [0.027, 0.027]
        #     )
        if self.model.process_material_frame_calibration(self.filter.pose):
            self.next_stage()
        return self.timer.PUBLISH_TIME.FALSE

    def stage_reconstruction(self) -> Timer.PUBLISH_TIME:
        self.log_debug(f"Reconstructing...")
        self.model.reconstruct(
            marker_pose=self.filter.pose,
        )
        new_time = time.time()
        self.log_info(
            f"Reconstructing... at rate: {1/(new_time-self.time):.2f} Hz. Done!"
        )
        self.time = new_time
        self.publish_messages()
        return self.timer.PUBLISH_TIME.TRUE

    def publish_messages(self) -> None:
        self.publish("pose", self.pose)
        self.publish("filtered_pose", self.filter.pose)
        self.publish(
            "lab_frame_transformation",
            self.model.lab_frame_transformation,
        )
        self.publish(
            "material_frame_transformation",
            self.model.material_frame_transformation,
        )
        self.publish("calibrated_pose", self.model.calibrated_pose)
        self.publish("input", self.model.input_pose)
        self.publish("position", self.model.result.position)
        self.publish("directors", self.model.result.directors)
        self.publish("kappa", self.model.result.kappa)

    def publish(
        self,
        publisher_key: str,
        data: np.ndarray,
    ) -> None:
        self.__publishers[publisher_key].release(data)
        self.log_debug(f"{self.__publishers[publisher_key]}")


def set_subsciption_topics(source: str, markers: int) -> Tuple[str]:
    if source.lower() == "vicon":
        subscription_topics = tuple(
            f"/vicon/br2_seg_{i}/br2_seg_{i}" for i in range(1, markers + 1)
        )
        # subscription_topics = (
        #     "/vicon/br2_seg_1/br2_seg_1",
        #     "/vicon/br2_seg_2/br2_seg_2",
        #     "/vicon/br2_seg_3/br2_seg_3",
        #     "/vicon/br2_seg_4/br2_seg_4",
        # )
    if source.lower() == "vicon_mock":
        subscription_topics = tuple(
            f"/vicon_mock/CrossSection_0_{i}/CrossSection_0_{i}"
            for i in range(markers)
        )
        # subscription_topics = (
        #     "/vicon_mock/CrossSection_0_0/CrossSection_0_0",
        #     "/vicon_mock/CrossSection_0_1/CrossSection_0_1",
        #     "/vicon_mock/CrossSection_0_2/CrossSection_0_2",
        #     "/vicon_mock/CrossSection_0_3/CrossSection_0_3",
        #     "/vicon_mock/CrossSection_0_4/CrossSection_0_4",
        #     "/vicon_mock/CrossSection_0_5/CrossSection_0_5",
        # )
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
@click.option(
    "--markers",
    type=int,
    default=4,
    help="Set the number of markers",
)
def main(log_level: str, source: str, markers: int):
    ros2_vicon.init()

    node = ReconstructionNode(
        subscription_topics=set_subsciption_topics(source, markers),
        log_level=log_level,
        # model_resource=ReconstructionModelResource(
        #     rotation_offset=-155.0,
        #     translation_offset=np.array([-0.02, -0.03, 0.0]),
        #     # translation_offset=np.array([0.0, 0.0, 0.0]),
        # ),
    )
    try:
        node.start()
    except KeyboardInterrupt:
        node.stop()
        ros2_vicon.shutdown()


if __name__ == "__main__":
    main()
