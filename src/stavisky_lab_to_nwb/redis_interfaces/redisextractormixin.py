import redis
import numpy as np
import scipy.signal as ssignal
from typing import Optional, Literal, Union


def safe_decode(string_or_bytes: Union[str, bytes], encoding="utf-8"):
    if isinstance(string_or_bytes, str):
        return string_or_bytes
    else:
        return str(string_or_bytes, encoding=encoding)


class RedisExtractorMixin:
    _client: redis.Redis
    
    def get_ids_and_timestamps(
        self,
        stream_name: str,
        frames_per_entry: int = 1,
        start_time: Optional[float] = None,
        sampling_frequency: Optional[float] = None,
        timestamp_source: Optional[Union[bytes, str]] = None,
        timestamp_unit: Optional[Literal["s", "ms", "us"]] = None,
        timestamp_encoding: Optional[Literal["str", "buffer"]] = None,
        timestamp_dtype: Optional[Union[str, type, np.dtype]] = None,
        chunk_size: int = 1000,
        smoothing_window: Union[int, Literal["max"]] = 1,
        smoothing_stride: int = 1,
        smoothing_causal_check: bool = False,
    ):
        # get stream len
        num_entries = self._client.xlen(stream_name)
        
        # get timestamps if already calculable
        if (start_time is not None) and (sampling_frequency is not None):
            timestamps = np.arange(num_entries * frames_per_entry, dtype=np.float64) / sampling_frequency
            build_timestamps = False
        else:
            build_timestamps = True
        
        # check args initialize buffer if timestamps need to be loaded
        if build_timestamps:
            assert (timestamp_source is not None)
            if safe_decode(timestamp_source).lower() == "redis": 
                timestamp_unit = "ms"
            else:
                assert (timestamp_encoding is not None)
                assert (timestamp_dtype is not None)
                assert (timestamp_unit is not None)
                if not isinstance(timestamp_source, bytes):
                    timestamp_source = bytes(timestamp_source, "utf-8")
            timestamps = np.full((num_entries * frames_per_entry,), np.nan, dtype=np.float64)
        timestamp_dtype = np.dtype(timestamp_dtype)
        
        # initialize variables for loop
        entry_ids = []
        last_ts = -np.inf
        count = 0
        
        # get first block of entries
        stream_entries = self._client.xrange(stream_name, count=chunk_size)
        
        # loop until all entries fetched
        while len(stream_entries) > 0:
            # store entry ids for indexing
            entry_ids += [entry[0] for entry in stream_entries]
            
            if build_timestamps:
                # build timestamps as specified
                for entry in stream_entries:
                    # use redis stream ids as timestamps
                    if safe_decode(timestamp_source).lower() == "redis":
                        # duplicate stream id as timestamp for each frame in entry
                        entry_timestamps = np.full((frames_per_entry,), float(safe_decode(entry[0]).split('-')[0]), dtype=np.float64)
                        assert entry_timestamps[0] >= last_ts # sanity check for monotonic increasing ts
                        last_ts = entry_timestamps[0]
                    else:
                        assert timestamp_source in entry[1].keys() # not all entries have to have the same keys
                        if timestamp_encoding == "str":
                            # pretty risky casting string to dtype, but some timestamps are stored as strings
                            entry_timestamps = np.full((frames_per_entry,), timestamp_dtype(safe_decode(entry[1][timestamp_source])), dtype=np.float64)
                        elif timestamp_encoding == "buffer":
                            # byte buffer reading, with duplicating timestamp if only one
                            entry_timestamps = np.frombuffer(entry[1][timestamp_source], dtype=timestamp_dtype)
                            if entry_timestamps.size == 1 and frames_per_entry > 1:
                                entry_timestamps = np.full((frames_per_entry,), entry_timestamps.item())
                            elif entry_timestamps.size != frames_per_entry:
                                raise AssertionError(
                                    f"Unexpected shape for timestamps of entry {entry[0]}. " + 
                                    f"Expected: (1,) or ({frames_per_entry},). Found: {entry_timestamps.shape}"
                                )
                        assert np.all(entry_timestamps > last_ts) # sanity check for monotonic increasing ts
                        last_ts = entry_timestamps[-1]
                    # add timestamps to array
                    timestamps[(count*frames_per_entry):((count+1)*frames_per_entry)] = entry_timestamps
                    count += 1
            
            # read next entries
            # '(' notation means exclusive range: see https://redis.io/commands/xrange/
            stream_entries = self._client.xrange(
                stream_name, min=b'(' + stream_entries[-1][0], count=chunk_size)
        
        # sanity check output
        assert len(entry_ids) == num_entries
        if build_timestamps:
            assert not np.any(np.isnan(timestamps))
        
        # convert to seconds
        if timestamp_unit == "ms":
            timestamps *= 1e-3
        elif timestamp_unit == "us":
            timestamps *= 1e-6
        
        # "smooth" timestamps if desired
        if smoothing_window != 1:
            timestamps = RedisExtractorMixin.smooth_timestamps(
                timestamps=timestamps,
                frames_per_entry=frames_per_entry,
                window_len=smoothing_window,
                stride_len=smoothing_stride,
                causal_check=smoothing_causal_check,
            )
        
        # compute frequency and period if necessary
        if sampling_frequency is None:
            sampling_frequency = 1. / np.median(np.diff(timestamps))
        
        # get start time if necessary
        if start_time is None:
            start_time = timestamps[0]
            timestamps -= start_time
        elif build_timestamps:
            timestamps -= start_time
        
        assert np.all(timestamps >= 0.) # sanity check
        
        return start_time, sampling_frequency, timestamps, entry_ids
    
    def smooth_timestamps(
        timestamps: np.ndarray,
        frames_per_entry: int,
        window_len: Union[int, Literal["max"]] = 1,
        stride_len: int = 1,
        causal_check: bool = False,
    ):
        # no smoothing if window_len of 1
        if window_len == 1:
            return timestamps
        elif window_len == "max":
            window_len = (len(timestamps) - frames_per_entry) * 2
        else:
            assert window_len > 1, '`window_len` must be either "max" or an integer >= 1'
        
        # take diffs
        ts_diff = np.diff(timestamps)
        ts_diff = ts_diff[(frames_per_entry-1):] # drop first entry since start time unknown
        
        # prepare sliding window kernel
        kernel = np.ones(window_len) / float(window_len)
        
        # pad data
        front_pad_len = window_len // 2
        back_pad_len = window_len // 2 - int((window_len % 2) == 0)
        ts_diff_padded = np.empty(ts_diff.size + front_pad_len + back_pad_len, dtype=np.float32)
        ts_diff_padded[front_pad_len:-back_pad_len] = ts_diff
        pad_view = np.flip(ts_diff.reshape(-1, frames_per_entry), axis=0)
        assert np.may_share_memory(pad_view, ts_diff)
        ts_diff_padded[:front_pad_len] = pad_view.flat[-front_pad_len:]
        ts_diff_padded[-back_pad_len:] = pad_view.flat[:back_pad_len]
        
        # perform convolution
        smoothed_diff = ssignal.convolve(
            ts_diff_padded,
            kernel,
            mode="valid",
        ).squeeze()
        assert len(smoothed_diff) == len(ts_diff) # sanity check sufficient padding
        
        # get strided values if desired
        # TODO: make more efficient? tons of unused computations in full convolve
        if stride_len > 1:
            smoothed_diff = smoothed_diff[(stride_len // 2)::stride_len]
            smoothed_diff = np.repeat(smoothed_diff, stride_len)
        # smoothed_diff = np.concatenate([smoothed_diff[:frames_per_entry], smoothed_diff], axis=0)
        
        # build timestamps and align with original end
        smoothed_timestamps = np.empty(timestamps.shape, dtype=np.float64)
        # (assume first entry is same as second)
        smoothed_timestamps[:frames_per_entry] = np.cumsum(smoothed_diff[:frames_per_entry])
        smoothed_timestamps[frames_per_entry:] = np.cumsum(smoothed_diff) + smoothed_timestamps[frames_per_entry-1]
        smoothed_timestamps += timestamps[-1] - smoothed_timestamps[-1]
        
        # optional: check that smoothed timestamps do not precede original
        # since that should not be possible
        if causal_check:
            assert np.all((timestamps - smoothed_timestamps) > 0.)
        
        return smoothed_timestamps
        