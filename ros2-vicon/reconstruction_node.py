import numpy as np
from dataclasses import dataclass, field
from typing import List

try:
    import rclpy
    from rclpy.node import Node
    from vicon_receiver.msg import Position
    from std_msgs.msg import Float32MultiArray, MultiArrayDimension  # For publishing numpy array
except ModuleNotFoundError:
    print('Could not import ROS2 modules. Make sure to source ROS2 workspace first.')
    import sys
    sys.exit(1)

@dataclass
class ReconstructionResult:
    number_of_elements: int
    position: np.ndarray = field(default=None, init=False)
    director: np.ndarray = field(default=None, init=False)

    def __post_init__(self):
        # Initialize position as a (3, number_of_elements) numpy array
        self.position = np.zeros((3, self.number_of_elements))
        # Initialize director as a (3, 3, number_of_elements) numpy array
        self.director = np.zeros((3, 3, self.number_of_elements))

class ReconstructionNode(Node):
    def __init__(
        self, 
        subscription_topics: List[str], 
        reconstruction_rate: float = 60.0,
        reconstructed_elements: int = 100,
    ):
        super().__init__('multi_subscriber_node')

        # Store subscription topics
        self.subscription_topics = subscription_topics

        # Reconstruction rate in Hz
        self.reconstruction_rate = reconstruction_rate

        # Subscribe to the first topic
        self.subscription_1 = self.create_subscription(
            msg_type=Position,
            topic=self.subscription_topics[0],
            callback=self.listener_callback_1,
            qos_profile=1,
        )
        self.subscription_1  # prevent unused variable warning

        # Subscribe to the second topic
        self.subscription_2 = self.create_subscription(
            msg_type=Position,
            topic=self.subscription_topics[1],
            callback=self.listener_callback_2,
            qos_profile=1,
        )
        self.subscription_2  # prevent unused variable warning

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

        # Store received messages (optional)
        self.data_1 = np.zeros(3)
        self.data_2 = np.zeros(3)
        self.data_1_frame_number = None
        self.data_2_frame_number = None

    def listener_callback_1(self, msg):
        self.data_1 = [msg.x_trans, msg.y_trans, msg.z_trans]
        self.data_1_frame_number = msg.frame_number

        self.get_logger().info(f'{msg.frame_number}')
        self.get_logger().info(f'  {msg.x_trans}')
        self.get_logger().info(f'  {msg.y_trans}')
        self.get_logger().info(f'  {msg.z_trans}')
        self.get_logger().info(f'  {msg.x_rot}')
        self.get_logger().info(f'  {msg.y_rot}')
        self.get_logger().info(f'  {msg.z_rot}')
        self.get_logger().info(f'  {msg.w}')

    def listener_callback_2(self, msg):
        self.data_2 = [msg.x_trans, msg.y_trans, msg.z_trans]
        self.data_2_frame_number = msg.frame_number

    def timer_callback(self):
        self.reconstruct()
        self.publish_position(self.result.position)
        self.publish_director(self.result.director)

    def reconstruct(self):
        # Calculate position
        self.result.position[:, 0] = self.data_1
        self.result.position[:, 1] = self.data_2
        # Calculate director
        self.result.director[:, :, 0] = np.array([self.data_1, self.data_2, self.data_1])
        self.result.director[:, :, 1] = np.array([self.data_2, self.data_1, self.data_2])
    
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
        self.get_logger().info(f'Published position: {position}')

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
        self.get_logger().info(f'Published director: {director}')


def main(args=None):
    rclpy.init(args=args)
    subscription_topics = [
        '/vicon/TestSubject_0/TestSubject_0',
        '/vicon/TestSubject_1/TestSubject_1'
    ]
    subscription_topics = [
        '/vicon_mock/TestSubject_0/CrossSection_0_0',
        '/vicon_mock/TestSubject_0/CrossSection_0_1'
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