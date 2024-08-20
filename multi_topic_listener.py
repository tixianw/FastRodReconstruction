import sys
import os
import time


import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from vicon_receiver.msg import Position

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from std_msgs.msg import String as StdString  # For publishing test_output

class MultiSubscriberNode(Node):
    def __init__(self):
        super().__init__('multi_subscriber_node')

        # Publisher to "test_topic"
        self.publisher_ = self.create_publisher(StdString, 'test_topic', 10)

        # Subscribe to the first topic
        self.subscription_1 = self.create_subscription(
            String,
            'topic_1',
            self.listener_callback_1,
            10)
        self.subscription_1  # prevent unused variable warning

        # Subscribe to the second topic
        self.subscription_2 = self.create_subscription(
            Image,
            'topic_2',
            self.listener_callback_2,
            10)
        self.subscription_2  # prevent unused variable warning

        # Store received messages (optional)
        self.data_1 = None
        self.data_2 = None

    def listener_callback_1(self, msg):
        self.data_1 = msg.data
        self.get_logger().info(f'Received message from topic_1: {self.data_1}')
        self.process_and_publish()

    def listener_callback_2(self, msg):
        self.data_2 = msg
        self.get_logger().info('Received image message from topic_2')
        self.process_and_publish()

    def process_and_publish(self):
        if self.data_1 and self.data_2:
            # Example processing: Combine data from both topics
            processed_data = f'Processed data: {self.data_1} with image info'
            msg = StdString()
            msg.data = processed_data

            # Publish the processed data
            self.publisher_.publish(msg)
            self.get_logger().info(f'Published: {processed_data}')

            # Optionally, reset data to avoid re-publishing the same data
            self.data_1 = None
            self.data_2 = None

def main(args=None):
    rclpy.init(args=args)
    node = MultiSubscriberNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()