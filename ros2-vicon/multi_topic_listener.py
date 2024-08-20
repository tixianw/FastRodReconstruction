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
from std_msgs.msg import Float32MultiArray, MultiArrayDimension  # For publishing numpy array

import numpy as np
from time import perf_counter

class MultiSubscriberNode(Node):
    def __init__(self):
        super().__init__('multi_subscriber_node')

        # Publisher to "numpy_array_topic"
        self.numpy_publisher = self.create_publisher(
            Float32MultiArray,
            'numpy_array_topic',
            1
        )

        # Subscribe to the first topic
        self.subscription_1 = self.create_subscription(
            Position,
            '/vicon/TestSubject_0/TestSubject_0',
            self.listener_callback_1,
            1
        )
        self.subscription_1  # prevent unused variable warning

        # Subscribe to the second topic
        self.subscription_2 = self.create_subscription(
            Position,
            '/vicon/TestSubject_1/TestSubject_1',
            self.listener_callback_2,
            1
        )
        self.subscription_2  # prevent unused variable warning

        # Store received messages (optional)
        self.data_1 = None
        self.data_2 = None
        self.data_1_frame_number = None
        self.data_2_frame_number = None
        self.start_time = perf_counter()

    def listener_callback_1(self, msg):
        self.data_1 = [msg.x_trans, msg.y_trans, msg.z_trans]
        self.data_1_frame_number = msg.frame_number
        self.get_logger().info(f'Received message from topic_1: {self.data_1}')
        self.process_and_publish()

    def listener_callback_2(self, msg):
        self.data_2 = [msg.x_trans, msg.y_trans, msg.z_trans]
        self.data_2_frame_number = msg.frame_number
        self.get_logger().info(f'Received message from topic_2: {self.data_2}')
        self.process_and_publish()

    def process_and_publish(self):
        if self.data_1 and self.data_2:
            # Calculate time elapsed since the first message was received
            elapsed_time = perf_counter() - self.start_time
            self.get_logger().info(f'sampling rate: {1/elapsed_time} Hz')

            # Create a 2D NumPy array
            np_array = np.array([
                self.data_1,
                self.data_2,
            ])

            # Create Float32MultiArray message
            msg = Float32MultiArray()

            # Set up MultiArrayLayout
            msg.layout.dim = [
                MultiArrayDimension(label="rows", size=np_array.shape[0], stride=np_array.shape[0] * np_array.shape[1]),
                MultiArrayDimension(label="cols", size=np_array.shape[1], stride=np_array.shape[1])
            ]
            msg.layout.data_offset = 0

            # Flatten the array and convert to list
            msg.data = np_array.flatten().tolist()

            # Publish the processed data
            self.numpy_publisher.publish(msg)
            self.get_logger().info(f'Published 2D NumPy array with {self.data_1_frame_number} and {self.data_2_frame_number}')
            self.get_logger().info(f'{np_array}')

            # Optionally, reset data to avoid re-publishing the same data
            self.data_1 = None
            self.data_2 = None
            # time.sleep(1)
        self.start_time = perf_counter()

def main(args=None):
    rclpy.init(args=args)
    node = MultiSubscriberNode()
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