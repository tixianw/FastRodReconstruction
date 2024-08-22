from typing import List, Optional, Tuple

from collections import defaultdict

import numpy as np

from neural_data_smoothing3D import pos_dir_to_input
from reconstruction import ReconstructionModel, ReconstructionResult
from ros2_vicon import NDArrayPublisher, PoseSubscriber

try:
    import rclpy
    from rclpy.node import Node
except ModuleNotFoundError:
    print(
        "Could not import ROS2 modules. Make sure to source ROS2 workspace first."
    )
    import sys

    sys.exit(1)


class ReconstructionModelResourceProtocol:
    data_file_name: Optional[str] = (None,)
    model_file_name: Optional[str] = (None,)


class ReconstructionNode(Node):
    def __init__(
        self,
        subscription_topics: Tuple[str],
        reconstruction_rate: float = 60.0,
        model_resource: Optional[ReconstructionModelResourceProtocol] = None,
    ):
        super().__init__("reconstruction_node")
        self.get_logger().info("Reconstruction node initializing...")

        self.__subscription_topics = subscription_topics
        self.__reconstruction_rate = reconstruction_rate

        # Initialize subscribers
        self.get_logger().info("- Subcribers initializing...")
        self.__subscribers: List[PoseSubscriber] = []
        for i, topic in enumerate(self.__subscription_topics):
            subscriber = PoseSubscriber(
                topic=topic,
                callback=self.subscriber_callback_closure(i),
                qos_profile=100,
                node=self,
            )
            self.__subscribers.append(subscriber)

        self.number_of_markers = len(self.__subscribers)

        # Initialize reconstruction model
        self.get_logger().info("- Reconstruction model initializing...")
        self.__reconstruction_model = (
            ReconstructionModel(
                data_file_name=model_resource.data_file_name,
                model_file_name=model_resource.model_file_name,
            )
            if model_resource
            else ReconstructionModel()
        )

        # Initialize publishers
        self.get_logger().info("- Publishers initializing...")
        self.__publishers = defaultdict(lambda: "No publisher")
        self.__publishers["position"] = NDArrayPublisher(
            topic="/reconstruction/position",
            shape=self.__reconstruction_model.result.position.shape,
            axis_labels=("position", "element"),
            qos_profile=100,
            node=self,
        )
        self.__publishers["directors"] = NDArrayPublisher(
            topic="/reconstruction/directors",
            shape=self.__reconstruction_model.result.directors.shape,
            axis_labels=("directors", "director_index", "element"),
            qos_profile=100,
            node=self,
        )

        self.__publishers["kappa"] = NDArrayPublisher(
            topic="/reconstruction/kappa",
            shape=self.__reconstruction_model.result.kappa.shape,
            axis_labels=("kappa", "element"),
            qos_profile=100,
            node=self,
        )

        # Create a timer for publishing at reconstruction_rate Hz
        self.__timer = self.create_timer(
            timer_period_sec=1 / self.__reconstruction_rate,
            callback=self.timer_callback,
        )

    @property
    def result(self) -> ReconstructionResult:
        return self.__reconstruction_model.result

    def subscriber_callback_closure(self, i: int):
        def subscriber_callback(msg):
            self.__subscribers[i].read(msg)
            self.get_logger().info(f"{self.__subscribers[i]}")

            # self.get_logger().info(f'{msg.frame_number}')
            # self.get_logger().info(f'  {msg.x_trans}')
            # self.get_logger().info(f'  {msg.y_trans}')
            # self.get_logger().info(f'  {msg.z_trans}')
            # self.get_logger().info(f'  {msg.x_rot}')
            # self.get_logger().info(f'  {msg.y_rot}')
            # self.get_logger().info(f'  {msg.z_rot}')
            # self.get_logger().info(f'  {msg.w}')

        return subscriber_callback

    def timer_callback(self):
        self.reconstruct()
        self.publish("position", self.result.position)
        self.publish("directors", self.result.directors)
        self.publish("kappa", self.result.kappa)

    def create_input_data(self) -> np.ndarray:
        # Create input data from the subscribers

        position_data = np.zeros((1, 3, self.number_of_markers - 1))
        director_data = np.zeros((1, 3, 3, self.number_of_markers - 1))
        base_position = self.__subscribers[0].message.position
        base_director = self.__subscribers[0].message.directors
        for i, subscriber in enumerate(self.__subscribers[1:]):
            position_data[0, :, i] = subscriber.message.position - base_position
            director_data[0, :, :, i] = subscriber.message.directors

        input_data = pos_dir_to_input(
            pos=position_data,
            dir=director_data,
        )
        return input_data

    def reconstruct(self):
        self.__reconstruction_model(self.create_input_data())

    def publish(self, publisher_key: str, data: np.ndarray):
        self.__publishers[publisher_key].publish(data)
        self.get_logger().info(f"{self.__publishers[publisher_key]}")


def main(args=None):
    rclpy.init(args=args)
    subscription_topics = (
        "/vicon/br2_seg_1/br2_seg_1",
        "/vicon/br2_seg_2/br2_seg_2",
        "/vicon/br2_seg_3/br2_seg_3",
        "/vicon/br2_seg_4/br2_seg_4",
    )
    # subscription_topics = (
    #     "/vicon_mock/CrossSection_0_0/CrossSection_0_0",
    #     "/vicon_mock/CrossSection_0_1/CrossSection_0_1",
    #     '/vicon_mock/CrossSection_0_2/CrossSection_0_2',
    #     '/vicon_mock/CrossSection_0_3/CrossSection_0_3',
    #     # '/vicon_mock/CrossSection_0_4/CrossSection_0_4',
    #     # '/vicon_mock/CrossSection_0_5/CrossSection_0_5',
    # )
    node = ReconstructionNode(
        subscription_topics=subscription_topics,
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
