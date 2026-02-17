import h5py
import numpy as np
from datetime import datetime
import os
import threading

from simdist.utils import paths

TIME_KEY = "time"
PROPRIO_KEY = "proprio_obs"
EXTERO_KEY = "extero_obs"
ACTS_KEY = "actions"
CMDS_KEY = "commands"


class HDF5EpisodeLogger:
    def __init__(self, cfg: dict, node):
        self.node = node
        self.enabled = cfg["enabled"]
        self.dataset_name = cfg["dataset_name"]
        self.file = None
        self.datasets = {}
        self.step = 0
        self.write_lock = threading.Lock()

    def open(self):
        if not self.enabled:
            return

        with self.write_lock:
            date = datetime.now().strftime("%Y-%m-%d")
            t = datetime.now().strftime("%H-%M-%S")
            raw_dir = paths.get_real_raw_data_dir(self.dataset_name)
            file_path = os.path.join(raw_dir, date, f"{t}.hdf5")
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            self.node.get_logger().info(f"Logging episode data to: {file_path}")

            self.file = h5py.File(file_path, "w")

            # Preallocate with chunked unlimited-size datasets
            self.datasets["time"] = self.file.create_dataset(
                "time", shape=(0,), maxshape=(None,), dtype="int64", chunks=True
            )
            self.datasets[PROPRIO_KEY] = self.file.create_dataset(
                PROPRIO_KEY,
                shape=(0, 0),
                maxshape=(None, None),
                dtype="float32",
                chunks=True,
            )
            self.datasets[EXTERO_KEY] = self.file.create_dataset(
                EXTERO_KEY,
                shape=(0, 0),
                maxshape=(None, None),
                dtype="float32",
                chunks=True,
            )
            self.datasets[ACTS_KEY] = self.file.create_dataset(
                ACTS_KEY,
                shape=(0, 0),
                maxshape=(None, None),
                dtype="float32",
                chunks=True,
            )
            self.datasets[CMDS_KEY] = self.file.create_dataset(
                CMDS_KEY,
                shape=(0, 0),
                maxshape=(None, None),
                dtype="float32",
                chunks=True,
            )

            self.step = 0

    def write(
        self,
        proprio_obs: np.ndarray,
        extero_obs: np.ndarray,
        action: np.ndarray,
        command: np.ndarray,
    ):
        if not self.enabled or self.file is None:
            return

        with self.write_lock:
            timestamp = self.node.get_clock().now().nanoseconds

            # First call: resize all datasets with known dims
            if self.step == 0:
                self.datasets[PROPRIO_KEY].resize((0, proprio_obs.size))
                self.datasets[EXTERO_KEY].resize((0, extero_obs.size))
                self.datasets[ACTS_KEY].resize((0, action.size))
                self.datasets[CMDS_KEY].resize((0, command.size))

            # Resize all datasets to append a new row
            for key in self.datasets:
                if key == "time":
                    self.datasets[key].resize((self.step + 1,))
                else:
                    self.datasets[key].resize(
                        (self.step + 1, self.datasets[key].shape[1])
                    )

            # Store data
            self.datasets[TIME_KEY][self.step] = timestamp
            self.datasets[PROPRIO_KEY][self.step] = proprio_obs.astype(np.float32)
            self.datasets[EXTERO_KEY][self.step] = extero_obs.astype(np.float32)
            self.datasets[ACTS_KEY][self.step] = action.astype(np.float32)
            self.datasets[CMDS_KEY][self.step] = command.astype(np.float32)

            self.step += 1

    def close(self):
        data_dict = {}

        with self.write_lock:
            if self.file:
                # Retrieve data
                for key, dataset in self.datasets.items():
                    data_dict[key] = dataset[:]

                # Close file and reset state
                self.file.close()
                self.file = None
                self.datasets = {}
                self.step = 0
            else:
                return None

        return data_dict
