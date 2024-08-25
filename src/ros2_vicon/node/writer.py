from typing import Tuple

from collections import deque
from dataclasses import dataclass
from threading import Event, Thread

import h5py


@dataclass
class HDF5Writer(Thread):
    """
    A simple thread that writes data from a buffer to a HDF5 file in chunks.
    """

    file_name: str
    keys: Tuple[str]
    chunk_size: int = 1000
    save_interval: float = 0.1

    def __post_init__(self):
        super().__init__()

        self.event = Event()

        self._buffer = {
            "position": deque(),
            "director": deque(),
            "kappa": deque(),
        }
        self._data_shape = {
            "position": (3, 4),
            "director": (3, 3, 4),
            "kappa": (1, 4),
        }

    def append_data(self, key, data):
        self._buffer[key].append(data)

    def write_buffer_to_hdf5(self, dset, data_chunk):
        dset.resize(dset.shape[0] + len(data_chunk), axis=0)
        dset[-len(data_chunk) :] = np.asarray(data_chunk)

    def is_buffer_exceeds_chunk_size(self):
        return len(self._buffer["position"]) >= self.chunk_size

    def run(self):
        with h5py.File(self.file_name, "w") as f:
            for key, shape in self._data_shape.items():
                f.create_dataset(
                    key,
                    (0,) + shape,
                    maxshape=(None,) + shape,
                    chunks=True,
                    dtype=np.float64,
                )

            while not self.event.is_set():
                if self.is_buffer_exceeds_chunk_size():
                    for key, dset in f.items():
                        self.write_buffer_to_hdf5(dset, self._buffer[key])
                        self._buffer[key].clear()
                time.sleep(self.save_interval)

    def stop(self):
        self.event.set()
        self.join()
