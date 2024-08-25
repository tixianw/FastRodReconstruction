from typing import Dict, List, Optional, Tuple, Union

import click
import numpy as np

try:
    import rclpy
    from rclpy.node import Node
except ModuleNotFoundError:
    print(
        "Could not import ROS2 modules. Make sure to source ROS2 workspace first."
    )
    import sys

    sys.exit(1)

from std_msgs.msg import (
    Float64MultiArray,
)  # TODO: Change this to match your message type

import h5py
import threading

from collections import deque
import time


class HDF5ReconstructionWriter(threading.Thread):
    """
    A simple thread that writes data from a buffer to an
    HDF5 file in chunks.
    """

    def __init__(self, file_name, chunk_size=1000, save_interval=0.1):
        """
        Parameters
        ----------
        file_name : str
            The name of the HDF5 file to write to.
        chunk_size : int
            The size of the chunks to write to the HDF5 file.
        save_interval : float
            The time to wait between saving chunks of data
            to the HDF5 file. Units are in seconds. (Default: 0.1)
        """
        super().__init__()
        self.stop_event = threading.Event()

        self.file_name = file_name
        self.chunk_size = chunk_size
        self.save_interval = save_interval

        self._buffer: dict[str, deque] = {
            "poses": deque(),
        }
        self._data_shape: dict[str, Tuple[int]] = {
            "poses": (3,),
        }

    def append_data(self, data):
        """
        Append data to the buffer.
        """
        self._buffer["poses"].append(data)
        # TODO: Add more data types here

    def write_buffer_to_hdf5(self, dset: h5py.Dataset, data_chunk: List):
        """
        Write a chunk of data to an HDF5 dataset.
        """
        dset.resize(dset.shape[0] + len(data_chunk), axis=0)
        dset[-len(data_chunk) :] = np.asarray(data_chunk)

    def is_buffer_exceeds_chunk_size(self) -> bool:
        """
        Check if the data in the buffer exceeds the chunk size.
        """
        return min([len(v) for v in self._buffer.values()]) >= self.chunk_size

    def is_buffer_empty(self) -> bool:
        """
        Check if the buffer is empty.
        """
        return all([len(v) == 0 for v in self._buffer.values()])

    def run(self):
        with h5py.File(self.file_name, "a") as f:
            # TODO: Might want to add and/or group the data
            dset_dictionary = {
                key: f.require_dataset(
                    key,
                    shape=(0, *self._data_shape[key]),
                    dtype=np.float64,
                    chunks=True,
                )
                for key in self._buffer.keys()
            }

            # Main loop
            while not self.stop_event.is_set():
                if not self.is_buffer_exceeds_chunk_size():
                    pass

                # Save data in chunks
                for key, buffer in self._buffer.items():
                    data_chunk = [
                        buffer.popleft() for _ in range(self.chunk_size)
                    ]
                    self.write_buffer_to_hdf5(dset_dictionary[key], data_chunk)
                time.sleep(self.save_interval)  # Non-zero sleep interval

            # Save any remaining data
            if not self.is_buffer_empty():
                for key, buffer in self._buffer.items():
                    self.write_buffer_to_hdf5(
                        dset_dictionary[key], list(buffer)
                    )

    def stop(self):
        self.stop_event.set()
        self.join()


class PoseHDF5Writer(Node):
    def __init__(self, log_level: str = "info"):
        super().__init__("pose_hdf5_writer")
        self.set_logger_level(log_level)
        self.get_logger().info("HDF5 writer node initializing...")

        self.subscription = self.create_subscription(
            Float64MultiArray,  # TODO: Adjust this to match your message type
            "/reconstruction/poses",
            self.listener_callback,
            10,
        )
        # TODO: Path to save h5
        self.writer = HDF5ReconstructionWriter("poses.h5")
        self.writer.start()

    def listener_callback(self, msg):
        self.writer.append_data(msg.data)

    def destroy_node(self):
        super().destory_node()
        self.writer.stop()


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
def main(log_level: str):
    rclpy.init()

    pose_hdf5_writer = PoseHDF5Writer(log_level=log_level)

    try:
        rclpy.spin(pose_hdf5_writer)
    except KeyboardInterrupt:
        pose_hdf5_writer.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
