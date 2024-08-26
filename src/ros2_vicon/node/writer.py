from typing import Dict

import time
from collections import deque
from dataclasses import dataclass
from threading import Event, Thread

import h5py
import numpy as np

from ros2_vicon.message.array import NDArrayMessage
from ros2_vicon.node import LoggerNode


@dataclass
class HDF5Writer(Thread):
    """
    A thread that writes message from a buffer to a HDF5 file in chunks.
    """

    file_name: str
    messages: Dict[str, NDArrayMessage]
    node: LoggerNode
    chunk_size: int = 1000
    writer_period_sec: float = 0.1

    def __post_init__(self):
        super().__init__()

        self.write_file_event = Event()
        self.write_file_event.set()

        self._buffer = {key: deque() for key in self.messages.keys()}

    def __hash__(self) -> int:
        return hash(id(self))

    def record(self, key: str, msg) -> None:
        self._buffer[key].append(self.messages[key].from_message(msg).to_hdf5())

    # BUG: no idea what this is doing
    def write_to_hdf5(self, f: h5py.File) -> None:
        for key, dset in f.items():
            dset.resize(dset.shape[0] + len(self._buffer[key]), axis=0)
            dset[-len(self._buffer[key]) :] = np.asarray(self._buffer[key])
            self._buffer[key].clear

    # TODO: bad implementation
    def is_filled(self):
        return len(self._buffer["position"]) >= self.chunk_size

    def run(self):
        with h5py.File(self.file_name, "w") as f:
            for key, message in self.messages.items():
                f.create_dataset(
                    name=key,
                    shape=(0,) + message.shape,
                    dtype=np.float32,
                    maxshape=(None,) + message.shape,
                    chunks=True,
                )

            while self.write_file_event.is_set():
                self.node.log_debug("Writing to HDF5 file...")
                if self.is_filled():
                    self.write_to_hdf5(f)
                time.sleep(self.writer_period_sec)

    def stop(self):
        self.write_file_event.clear()
        self.join()
