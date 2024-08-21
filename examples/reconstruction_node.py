import sys
sys.path.append('../')

import numpy as np
from collections import defaultdict
from typing import Tuple
from reconstruction import ReconstructionResult, ReconstructionModel
from ros2_vicon import PoseMsg, PoseSubscriber, NDArrayMessage

try:
    import rclpy
    from rclpy.node import Node
    from vicon_receiver.msg import Position as Pose
    from std_msgs.msg import Float32MultiArray, MultiArrayDimension  # For publishing numpy array
except ModuleNotFoundError:
    print('Could not import ROS2 modules. Make sure to source ROS2 workspace first.')
    import sys
    sys.exit(1)


class ReconstructionNode(Node):
    def __init__(
        self, 
        subscription_topics: Tuple[str], 
        reconstruction_rate: float = 60.0,
        reconstructed_elements: int = 100,
        model: ReconstructionModel = None,
    ):
        super().__init__('reconstruction_node')
        self.get_logger().info('Reconstruction node initializing...')

        self.__subscription_topics = subscription_topics
        self.__reconstruction_rate = reconstruction_rate
        self.model = model

        # Initialize subscribers
        self.get_logger().info('- Subcribers initializing...')
        self.__subscribers = []
        for i, topic in enumerate(self.__subscription_topics):
            subscriber = PoseSubscriber(
                topic=topic,
                data=PoseMsg(),
                subscription=self.create_subscription(
                    msg_type=Pose,
                    topic=topic,
                    callback=self.subscriber_callback_closure(i),
                    qos_profile=100,
                )
            )
            self.__subscribers.append(subscriber)

        # Initialize publishers
        self.get_logger().info('- Publishers initializing...')
        self.__publishing_topics = {
            'position': NDArrayMessage(
                topic='/reconstruction/position',
                shape=(3, reconstructed_elements), 
                axis_labels=('position', 'element')
            ),
            'directors': NDArrayMessage(
                topic='/reconstruction/directors',
                shape=(3, 3, reconstructed_elements), 
                axis_labels=('directors', 'director_index', 'element')
            ),
        }
        self.__publishers = defaultdict(lambda: "No publisher")
        for key, message in self.__publishing_topics.items():
            publisher = self.create_publisher(
                msg_type=message.TYPE,
                topic=message.topic,
                qos_profile=1,
            )
            self.__publishers[key] = publisher

        # Create a timer for publishing at reconstruction_rate Hz
        self.__timer = self.create_timer(
            timer_period_sec=1/self.__reconstruction_rate, 
            callback=self.timer_callback,
        )

        self.result = ReconstructionResult(
            number_of_elements=reconstructed_elements,
        )


    def subscriber_callback_closure(self, i: int):
        def subscriber_callback(msg):
            self.__subscribers[i].data.frame_number = msg.frame_number
            self.__subscribers[i].data.position = [msg.x_trans, msg.y_trans, msg.z_trans]
            self.__subscribers[i].data.quaternion = [msg.x_rot, msg.y_rot, msg.z_rot, msg.w]
            
            self.get_logger().info(f'{self.__subscribers[i]}')
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
        self.publish_position(self.result.position)
        self.publish_director(self.result.directors)
        # self.get_logger().info(f'Published reconstruction result: {self.result}')

    def reconstruct(self):
        # Calculate position
        self.result.position[:, 0] = self.__subscribers[0].data.position
        self.result.position[:, 1] = self.__subscribers[1].data.position
        # Calculate director
        self.result.directors[:, :, 0] = self.__subscribers[0].data.directors
        self.result.directors[:, :, 1] = self.__subscribers[1].data.directors

    def publish_position(self, position: np.ndarray):
        # Create Float32MultiArray message
        msg = Float32MultiArray()

        # Set up MultiArrayLayout
        msg.layout.dim = [
            MultiArrayDimension(
                label="position", 
                size=position.shape[0], 
                stride=position.shape[0] * position.shape[1]
            ),
            MultiArrayDimension(
                label="element", 
                size=position.shape[1], 
                stride=position.shape[1]
            )
        ]
        msg.layout.data_offset = 0

        # Flatten the array and convert to list
        msg.data = position.flatten().tolist()

        # Publish the processed data
        self.__publishers['position'].publish(msg)

    def publish_director(self, director: np.ndarray):
        # Create Float32MultiArray message
        msg = Float32MultiArray()

        # Set up MultiArrayLayout
        msg.layout.dim = [
            MultiArrayDimension(
                label="director", 
                size=director.shape[0], 
                stride=director.shape[0] * director.shape[1] * director.shape[2]
            ),
            MultiArrayDimension(
                label="director_index", 
                size=director.shape[1], 
                stride=director.shape[1] * director.shape[2]
            ),
            MultiArrayDimension(
                label="element", 
                size=director.shape[2], 
                stride=director.shape[2]
            )
        ]
        msg.layout.data_offset = 0

        # Flatten the array and convert to list
        msg.data = director.flatten().tolist()

        # Publish the processed data
        self.__publishers['directors'].publish(msg)


def main(args=None):
    rclpy.init(args=args)
    subscription_topics = (
        '/vicon/TestSubject_0/TestSubject_0',
        '/vicon/TestSubject_1/TestSubject_1'
    )
    subscription_topics = (
        '/vicon_mock/CrossSection_0_0/CrossSection_0_0',
        '/vicon_mock/CrossSection_0_1/CrossSection_0_1',
        '/vicon_mock/CrossSection_0_2/CrossSection_0_2',
        '/vicon_mock/CrossSection_0_3/CrossSection_0_3',
        '/vicon_mock/CrossSection_0_4/CrossSection_0_4',
        '/vicon_mock/CrossSection_0_5/CrossSection_0_5',
    )
    node = ReconstructionNode(
        subscription_topics=subscription_topics,
        reconstructed_elements=len(subscription_topics),
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        # Destroy the node explicitly
        # (optional - otherwise it will be done automatically
        # when the garbage collector destroys the node object)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()