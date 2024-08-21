import sys
sys.path.append('../')

import numpy as np
from dataclasses import dataclass, field
from typing import List
from reconstruction import ReconstructionResult, ReconstructionModel
from ros2_vicon import PositionMsg, PositionSubscriber

try:
    import rclpy
    from rclpy.node import Node
    from vicon_receiver.msg import Position
    from std_msgs.msg import Float32MultiArray, MultiArrayDimension  # For publishing numpy array
except ModuleNotFoundError:
    print('Could not import ROS2 modules. Make sure to source ROS2 workspace first.')
    import sys
    sys.exit(1)


class ReconstructionNode(Node):
    def __init__(
        self, 
        subscription_topics: List[str], 
        reconstruction_rate: float = 60.0,
        reconstructed_elements: int = 100,
        model: ReconstructionModel = None,
    ):
        super().__init__('reconstruction_node')

        self.subscription_topics = subscription_topics
        self.reconstruction_rate = reconstruction_rate
        self.model = model

        # Create subscribers for each topic
        self.subscribers = []
        for i, topic in enumerate(self.subscription_topics):
            subscriber = PositionSubscriber(
                topic=topic,
                data=PositionMsg(),
                subscription=self.create_subscription(
                    msg_type=Position,
                    topic=topic,
                    callback=self.subscriber_callback_closure(i),
                    qos_profile=100,
                )
            )
            self.subscribers.append(subscriber)

        # Publisher to "reconstruction/position" topic
        self.publisher_position = self.create_publisher(
            msg_type=Float32MultiArray,
            topic='/reconstruction/position',
            qos_profile=1,
        )

        # Publisher to "reconstruction/director" topic
        self.publisher_director = self.create_publisher(
            msg_type=Float32MultiArray,
            topic='/reconstruction/director',
            qos_profile=1,
        )

        # Create a timer for publishing at reconstruction_rate Hz
        self.timer = self.create_timer(
            timer_period_sec=1/self.reconstruction_rate, 
            callback=self.timer_callback,
        )

        self.result = ReconstructionResult(
            number_of_elements=reconstructed_elements,
        )

    def subscriber_callback_closure(self, i: int):
        def subscriber_callback(msg):
            self.subscribers[i].data.frame_number = msg.frame_number
            self.subscribers[i].data.position = [msg.x_trans, msg.y_trans, msg.z_trans]
            self.subscribers[i].data.quaternion = [msg.x_rot, msg.y_rot, msg.z_rot, msg.w]
            
            self.get_logger().info(f'{self.subscribers[i]}')
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
        self.publish_director(self.result.director)
        # self.get_logger().info(f'Published reconstruction result: {self.result}')

    def reconstruct(self):
        # Calculate position
        self.result.position[:, 0] = self.subscribers[0].data.position
        self.result.position[:, 1] = self.subscribers[1].data.position
        # Calculate director
        self.result.director[:, :, 0] = np.array(
            [self.subscribers[0].data.position, self.subscribers[1].data.position, self.subscribers[0].data.position]
        )
        self.result.director[:, :, 1] = np.array(
            [self.subscribers[1].data.position, self.subscribers[0].data.position, self.subscribers[1].data.position]
        )
    
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
        self.publisher_position.publish(msg)

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
        self.publisher_director.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    subscription_topics = [
        '/vicon/TestSubject_0/TestSubject_0',
        '/vicon/TestSubject_1/TestSubject_1'
    ]
    subscription_topics = [
        '/vicon_mock/CrossSection_0_0/CrossSection_0_0',
        '/vicon_mock/CrossSection_0_1/CrossSection_0_1'
    ]
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