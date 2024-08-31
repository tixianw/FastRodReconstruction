import h5py
import numpy as np


def read_data_from_file(file_path, start_index, end_index):
    with h5py.File(file_path, "r") as f:
        start_index = 1100
        end_index = 2300
        time_length = end_index - start_index + 1
        input_ = np.array(f["input"])
        marker_length = input_.shape[3]
        data_tensor = np.zeros((time_length, 9, marker_length))
        for n in range(marker_length):
            data_tensor[:, :3, n] = input_[
                start_index : end_index + 1, :3, 3, n
            ]
            data_tensor[:, 3:6, n] = input_[
                start_index : end_index + 1, 0, :3, n
            ]
            data_tensor[:, 6:, n] = input_[
                start_index : end_index + 1, 2, :3, n
            ]
    return data_tensor


def main():
    data_tensor = read_data_from_file(
        file_path="experiment.h5",
        start_index=1100,
        end_index=2300,
    )
    print(data_tensor)
    print(data_tensor.shape)


if __name__ == "__main__":
    main()
