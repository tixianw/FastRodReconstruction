from typing import Dict

import time
from collections import defaultdict, deque
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
    chunk_size: int = 100
    writer_period_sec: float = 0.1
    verbose: bool = False

    def __post_init__(self):
        super().__init__()

        self.writer_event = Event()
        self.writer_event.set()

        self._buffer = {key: deque() for key in self.messages.keys()}

        self.log_verbose: callable = (
            self.node.log_info if self.verbose else self.node.log_debug
        )

    def __hash__(self) -> int:
        return hash(id(self))

    def record(self, key: str, msg) -> None:
        self._buffer[key].append(self.messages[key].from_message(msg).to_hdf5())

    def write_to_hdf5(self, f: h5py.File) -> None:
        chunk_size_dict = defaultdict(float)
        for key in self._buffer.keys():
            chunk_size = min(len(self._buffer[key]), self.chunk_size)
            self.node.log_debug(
                f"Writing chunk with size {chunk_size} to HDF5 file with key {key}..."
            )
            data_chunk = [
                self._buffer[key].popleft() for _ in range(chunk_size)
            ]

            dset = f[key]
            dset.resize(dset.shape[0] + chunk_size, axis=0)
            dset[-chunk_size:] = np.asarray(data_chunk)
            chunk_size_dict[key] = chunk_size

        if len(list(set(list(chunk_size_dict.values())))) == 1:
            self.log_verbose(
                f"Writing chunk to HDF5 file... with same chunk size {chunk_size}. Done!"
            )
        else:
            self.log_verbose(
                f"Writing chunk to HDF5 file... with different chunk size. Done!"
            )
            for key, chunk_size in chunk_size_dict.items():
                self.log_verbose(f"chunk size of {key}: {chunk_size}")

    def is_filled(self) -> bool:
        return (
            min([len(self._buffer[key]) for key in self._buffer.keys()])
            >= self.chunk_size
        )

    def is_empty(self) -> bool:
        return all([len(self._buffer[key]) == 0 for key in self._buffer.keys()])

    def run(self):
        with h5py.File(self.file_name, "w") as f:
            for key, message in self.messages.items():
                f.require_dataset(
                    name=key,
                    shape=(0,) + message.shape,
                    dtype=np.float32,
                    maxshape=(None,) + message.shape,
                    chunks=True,
                )

            while self.writer_event.is_set():
                if self.is_filled():
                    self.write_to_hdf5(f)
                time.sleep(self.writer_period_sec)

            if self.is_empty():
                self.write_to_hdf5(f)

    def stop(self):
        self.writer_event.clear()
        self.join()
