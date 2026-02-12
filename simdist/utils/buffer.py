import numpy as np
import os
from copy import deepcopy
from typing import Optional


class RingBuffer:
    def __init__(self, maxlen: int, shape: tuple, dtype=np.float32):
        """
        Args:
            maxlen: number of items in the buffer
            shape: shape of each item
            dtype: numpy dtype
        """
        self.maxlen = maxlen
        self.shape = shape
        self.buffer = np.zeros((maxlen, *shape), dtype=dtype)
        self.idx = 0
        self.full = False

    def append(self, item: np.ndarray):
        self.buffer[self.idx] = item
        self.idx = (self.idx + 1) % self.maxlen
        if self.idx == 0:
            self.full = True

    def extend(self, items: np.ndarray):
        """
        Appends multiple items to the buffer.
        Args:
            items: array of shape (batch_size, *item_shape)
        """
        num_items = items.shape[0]
        assert items.shape[1:] == self.shape

        # If input is larger than buffer, only keep the tail end
        if num_items > self.maxlen:
            items = items[-self.maxlen :]
            num_items = self.maxlen

        space_left = self.maxlen - self.idx
        if num_items <= space_left:
            self.buffer[self.idx : self.idx + num_items] = items
            self.idx = (self.idx + num_items) % self.maxlen  # Wrap to 0 if exactly full
        else:
            first_chunk = space_left
            second_chunk = num_items - first_chunk
            self.buffer[self.idx :] = items[:first_chunk]
            self.buffer[:second_chunk] = items[first_chunk:]
            self.idx = second_chunk

        if num_items >= self.maxlen or (self.idx == 0 and num_items > 0):
            self.full = True

    def get(self):
        """Returns the buffer in correct time order (oldest → newest)."""
        if self.full:
            return np.concatenate(
                [self.buffer[self.idx :], self.buffer[: self.idx]], axis=0
            )
        else:
            return self.buffer[: self.idx]

    def fill(self, item: np.ndarray):
        """Fills the entire buffer with a single item."""
        assert (
            item.shape == self.shape
        ), f"Expected item shape {self.shape}, got {item.shape}"
        self.buffer = np.tile(item[None], (self.maxlen, *([1] * len(self.shape))))
        self.idx = 0
        self.full = True

    def clear(self):
        """Clears the buffer, resetting it to an empty state."""
        self.buffer = np.zeros_like(self.buffer)
        self.idx = 0
        self.full = False

    def length(self):
        """Returns the current length of the buffer."""
        if self.full:
            return self.maxlen
        return self.idx


class MultiRingBuffer:
    def __init__(self, maxlen: int, example_item: dict[str, np.ndarray]):
        """
        Args:
            maxlen: number of items in the buffer
            example_item: dict of np arrays to define shapes and dtypes
        """
        self.maxlen = maxlen
        self.keys = list(example_item.keys())
        self.buffers = [
            RingBuffer(maxlen, value.shape, value.dtype)
            for value in example_item.values()
        ]
        self.episode_starts = RingBuffer(maxlen, shape=(), dtype=bool)

    def append(self, item: dict[str, np.ndarray], episode_start: bool = False):
        for k, buf in zip(self.keys, self.buffers):
            buf.append(item[k])
        self.episode_starts.append(np.array(episode_start, dtype=bool))

    def extend(self, items: dict[str, np.ndarray], episode_starts: np.ndarray):
        for k, buf in zip(self.keys, self.buffers):
            buf.extend(items[k])
        self.episode_starts.extend(episode_starts)

    def get(self) -> dict[str, np.ndarray]:
        return {k: buf.get() for k, buf in zip(self.keys, self.buffers)}

    def fill(self, item: dict[str, np.ndarray]):
        for k, buf in zip(self.keys, self.buffers):
            buf.fill(item[k])
        self.episode_starts.fill(np.array(False, dtype=bool))

    def clear(self):
        for buf in self.buffers:
            buf.clear()
        self.episode_starts.clear()

    def length(self) -> int:
        return min(buf.length() for buf in self.buffers)

    def save(self, path: Optional[str] = None):
        self._save_latest_arrays()
        if path is None or path == "":
            return
        for key, array in self._latest_arrays.items():
            np.save(os.path.join(path, f"{key}.npy"), array)
        np.save(os.path.join(path, "episode_starts.npy"), self._latest_episode_starts)

    def load(self, path: Optional[str] = None):
        if path is None or path == "":
            arrays = self._latest_arrays
            episode_starts = self._latest_episode_starts
        else:
            arrays = {}
            for key in self.keys:
                arrays[key] = np.load(os.path.join(path, f"{key}.npy"))
            episode_starts = np.load(os.path.join(path, "episode_starts.npy"))

        self.clear()
        self.extend(arrays, episode_starts)

    def _save_latest_arrays(self):
        self._latest_arrays = deepcopy(self.get())
        self._latest_episode_starts = deepcopy(self.episode_starts.get())

    def sample_sequences(
        self,
        batch_size: int,
        seq_len: int,
        include_latest: int = 0,
        sample_from_latest: int | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Samples a batch of contiguous sequences (B, T, dim) for each key.

        Args:
            batch_size: B
            seq_len: T
            include_latest: ensure up to this many sequences end at the most recent
                timesteps. For k>0, the first k sequences in the batch will end at
                N-1, N-2, ..., N-k respectively (clipped to valid range).
            sample_from_latest: if specified, only sample sequences that start
                within the last `sample_from_latest` timesteps.

        Returns:
            dict of {key: np.ndarray of shape (B, T, dim, ...)}
        """

        N = self.length()
        if N < seq_len:
            return None  # Not enough data to sample a sequence

        arrays = self.get()
        resets = self.episode_starts.get()  # Shape (N,)

        max_start = N - seq_len

        # 3. Create a mask of valid start indices
        # Default: all starts up to max_start are valid
        valid_starts_mask = np.ones(max_start + 1, dtype=bool)

        # If 'sample_from_latest' is set, mask out old data
        if sample_from_latest is not None:
            sample_from_latest = max(0, sample_from_latest)
            min_start = max(0, max_start - sample_from_latest)
            valid_starts_mask[:min_start] = False

        # 4. Mask out invalid sequences that cross a reset
        # If resets[k] is True, no sequence can cross index k.
        # This invalidates start indices in range [k - seq_len + 1, k - 1].
        reset_indices = np.where(resets)[0]

        for r_idx in reset_indices:
            # We want to forbid any sequence that CONTAINS r_idx
            # but does not START at r_idx
            # Valid sequence indices: [start, start + 1, ... start + seq_len - 1]
            # If r_idx is in that range (exclusive of start), it's invalid.
            # So invalid starts are: r_idx - (seq_len - 1) ... r_idx - 1
            bad_start_low = max(0, r_idx - seq_len + 1)
            bad_start_high = r_idx  # Exclusive

            if bad_start_low < bad_start_high:
                # We clip high to ensure we don't index out of bounds of the mask
                clip_high = min(bad_start_high, max_start + 1)
                valid_starts_mask[bad_start_low:clip_high] = False

        # Get all valid indices
        valid_indices = np.where(valid_starts_mask)[0]

        if len(valid_indices) == 0:
            print(
                "Warning: No valid sequences found "
                "(buffer might be too small or too many resets)."
            )
            return None

        # 5. Sampling Logic
        starts = np.empty((batch_size,), dtype=np.int64)

        # Handle 'include_latest': prioritize recent valid sequences
        k = int(max(0, min(include_latest, batch_size)))
        num_forced = 0

        if k > 0:
            # Try to pick the k latest VALID start indices
            # We look at the end of the valid_indices array
            if len(valid_indices) >= k:
                starts[:k] = valid_indices[-k:]
                num_forced = k
            else:
                # If we don't have k valid sequences total, take all of them
                starts[: len(valid_indices)] = valid_indices
                num_forced = len(valid_indices)

        # Fill the rest randomly from valid_indices
        if num_forced < batch_size:
            random_picks = np.random.randint(
                0, len(valid_indices), size=(batch_size - num_forced,)
            )
            starts[num_forced:] = valid_indices[random_picks]

        # Construct batch
        t_idx = np.arange(seq_len)[None, :]
        s_idx = starts[:, None]
        idx = s_idx + t_idx

        out: dict[str, np.ndarray] = {}
        for key in arrays.keys():
            out[key] = arrays[key][idx]

        return out
