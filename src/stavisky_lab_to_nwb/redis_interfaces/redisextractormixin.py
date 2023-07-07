import redis
import numpy as np
import scipy.signal as ssignal
from typing import Optional, Literal, Union
from warnings import warn


def safe_decode(string_or_bytes: Union[str, bytes], encoding="utf-8"):
    """Convenience method to decode var that can be either str or bytes"""
    if isinstance(string_or_bytes, str):
        return string_or_bytes
    else:
        return str(string_or_bytes, encoding=encoding)


class RedisExtractorMixin:
    """Class mixin for recording/sorting extractors for Redis,
    handles getting entry IDs and timestamps"""

    _client: redis.Redis

    def get_ids_and_timestamps(
        self,
        stream_name: str,
        frames_per_entry: int = 1,
        sampling_frequency: Optional[float] = None,
        timestamp_source: Optional[Union[bytes, str]] = None,
        timestamp_conversion: float = 1.,
        timestamp_encoding: Optional[Literal["str", "buffer"]] = None,
        timestamp_dtype: Optional[Union[str, type, np.dtype]] = None,
        chunk_size: int = 1000,
        smoothing_window: Union[int, Literal["max"]] = 1,
        smoothing_stride: int = 1,
        smoothing_causal_check: bool = False,
    ):
        """Main class function, gets entry IDs for stream indexing
        and computes timestamps for frames

        Parameters
        ----------
        stream_name : str
            Name of the Redis data stream to get IDs and timestamps for
        frames_per_entry : int, default: 1
            Number of frames (i.e. a single time point) contained within
            each Redis stream entry
        sampling_frequency : float, optional
            The sampling frequency of the data in Hz. If both `start_time` and
            `sampling_frequency` are provided, the timestamps are directly
            constructed from the two. If `sampling_frequency` is not provided,
            it is estimated with the median inter-sample time difference
        timestamp_source : bytes or str, optional
            The source of the timestamp information in the Redis stream. If
            the timestamp source is "redis", the entry IDs are used as timestamps.
            Otherwise, the timestamp source is assumed to be a data key present
            in each entry. If not provided, `start_time` and `sampling_frequency`
            must be specified for timestamps to be constructed
        timestamp_conversion : float, default: 1
            If `timestamp_source` is a Redis entry data key, then the user should
            provide the conversion factor needed to scale the timestamps to seconds
        timestamp_encoding : {"string", "buffer"}, optional
            If `timestamp_source` is a Redis entry data key, then how the
            timestamp is stored must be specified. If the encoding is "string",
            then the timestamp is treated as a human-readable string of some
            numeric type. If the encoding is "buffer", then the timestamp is
            treated as a raw byte buffer of some numeric type
        timestamp_dtype : str, type, or numpy.dtype, optional
            If `timestamp_source` is a Redis entry data key, then the data
            type of the timestamp must be specified. The provided data type
            is assumed to be a numeric type recognized by numpy, e.g. int8,
            float, float32, etc.
        chunk_size : int, default: 1000
            The number of Redis entries to read at once. Lower values reduce
            memory usage but increase runtime
        smoothing_window : int, default: 1
            The width of the rectangular window used to smooth irregular
            sampling frequencies. The default width of 1 performs no
            smoothing. See `RedisExtractorMixin.smooth_timestamps()`
            for more details
        smoothing_stride: int, default: 1
            The convolution stride used when smoothing timestamps.
            See `RedisExtractorMixin.smooth_timestamps()` for more details
        smoothing_causal_check : bool, default: False
            If True, smoothed timestamps are enforced to precede their
            corresponding un-smoothed timestamps. See
            `RedisExtractorMixin.smooth_timestamps()` for more details.
            TODO: potentially remove

        Returns
        -------
        start_time : float
            The provided start time, or the start time inferred from
            the timestamps
        sampling_frequency : float
            The provided sampling frequency, or the sampling frequency
            inferred from the timestamps
        timestamps : numpy.ndarray
            An array containing the timestamps either directly constructed
            from provided information or extracted from Redis, with optional
            smoothing
        entry_ids : list of bytes
            A list containing the entry ID for each entry in the Redis stream
            in order
        """
        # get stream len
        num_entries = self._client.xlen(stream_name)

        # get timestamps if already calculable
        if (sampling_frequency is not None):
            timestamps = np.arange(num_entries * frames_per_entry, dtype=np.float64) / sampling_frequency
            build_timestamps = False
        else:
            build_timestamps = True

        # check args initialize buffer if timestamps need to be loaded
        if build_timestamps:
            assert timestamp_source is not None
            if safe_decode(timestamp_source).lower() == "redis":
                timestamp_conversion = 1e-3
            else:
                assert timestamp_encoding is not None
                assert timestamp_dtype is not None
                assert timestamp_conversion is not None
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
                        entry_timestamps = np.full(
                            (frames_per_entry,), float(safe_decode(entry[0]).split("-")[0]), dtype=np.float128
                        )
                        assert entry_timestamps[0] >= last_ts  # sanity check for monotonic increasing ts
                        last_ts = entry_timestamps[0]
                    else:
                        assert timestamp_source in entry[1].keys()  # not all entries have to have the same keys
                        if timestamp_encoding == "str":
                            # pretty risky casting string to dtype, but some timestamps are stored as strings
                            entry_timestamps = np.full(
                                (frames_per_entry,),
                                timestamp_dtype(safe_decode(entry[1][timestamp_source])),
                                dtype=np.float128,
                            )
                        elif timestamp_encoding == "buffer":
                            # byte buffer reading, with duplicating timestamp if only one
                            entry_timestamps = np.frombuffer(entry[1][timestamp_source], dtype=timestamp_dtype)
                            if entry_timestamps.size == 1 and frames_per_entry > 1:
                                entry_timestamps = np.full((frames_per_entry,), entry_timestamps.item())
                            elif entry_timestamps.size != frames_per_entry:
                                raise AssertionError(
                                    f"Unexpected shape for timestamps of entry {entry[0]}. "
                                    + f"Expected: (1,) or ({frames_per_entry},). Found: {entry_timestamps.shape}"
                                )
                            entry_timestamps = entry_timestamps.astype(np.float128, copy=False)
                        assert np.all(entry_timestamps > last_ts)  # sanity check for monotonic increasing ts
                        last_ts = entry_timestamps[-1]
                    # convert to seconds and subtract start time
                    entry_timestamps *= timestamp_conversion
                    # add timestamps to array
                    timestamps[(count * frames_per_entry) : ((count + 1) * frames_per_entry)] = entry_timestamps
                    count += 1

            # read next entries
            # '(' notation means exclusive range: see https://redis.io/commands/xrange/
            stream_entries = self._client.xrange(stream_name, min=b"(" + stream_entries[-1][0], count=chunk_size)

        # sanity check output
        assert len(entry_ids) == num_entries
        if build_timestamps:
            assert not np.any(np.isnan(timestamps))

        # "smooth" timestamps if desired and
        if build_timestamps:
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
            sampling_frequency = 1.0 / np.median(np.diff(timestamps))

        assert np.all(timestamps >= 0.0)  # sanity check

        return sampling_frequency, timestamps, entry_ids

    def smooth_timestamps(
        timestamps: np.ndarray,
        frames_per_entry: int,
        window_len: Union[int, Literal["max"]] = 1,
        stride_len: int = 1,
        causal_check: bool = False,
    ):
        """Class function that smooths irregular timestamps.
        Internally, convolves inter-sample differences with a
        sliding rectangular window, then reconstructs timestamps
        with cumulative sum of smoothed differences

        Parameters
        ----------
        timestamps : numpy.ndarray
            An array of timestamps before smoothing
        frames_per_entry : int
            Number of frames per Redis stream entry, to exclude the
            first entry from timestamp smoothing. This is done because
            there is no calculable sampling period preceding the first
            entry. Instead, we assume the sampling period is equal to
            that of the second entry
        window_len : int or "max", default: 1
            Width of the sliding rectangular window used to smooth
            inter-sample differences. If window_len = "max", uses
            the maximum possible window size, which gives a constant
            sampling frequency for the entire segment. If window_len = 1,
            no smoothing is performed
        stride_len : int, default: 1
            Stride of the sliding window convolution. If stride = 1,
            normal convolution is used. If stride = n > 1, every nth value
            of the convolution, starting with the n/2th value, is taken as
            the sampling period for that block of n timestamps. For example,
            if there are multiple frames per entry, and you want to have a
            constant sampling frequency within the entry, but for each
            entry to have a potentially different sampling frequency, then
            you can set stride_len = frames_per_entry
        causal_check : bool, default: False
            Optional check that all smoothed timestamps precede their
            corresponding original timestamps. Since original timestamps are
            often the write times of a frame/entry, it should not be possible
            for the "true" timestamps to follow the recorded timestamps
        """
        # no smoothing if window_len of 1
        if window_len == 1:
            return timestamps
        elif window_len == "max":
            window_len = (len(timestamps) - frames_per_entry) * 2
        else:
            assert window_len > 1, '`window_len` must be either "max" or an integer >= 1'

        # take diffs
        ts_diff = np.diff(timestamps)
        ts_diff = ts_diff[(frames_per_entry - 1) :]  # drop first entry since start time unknown

        # prepare sliding window kernel
        kernel = np.ones(window_len) / float(window_len)

        # pad data
        front_pad_len = window_len // 2
        back_pad_len = window_len // 2 - int((window_len % 2) == 0)
        ts_diff_padded = np.empty(ts_diff.size + front_pad_len + back_pad_len, dtype=np.float64)
        ts_diff_padded[front_pad_len:-back_pad_len] = ts_diff
        pad_view = np.flip(ts_diff.reshape(-1, frames_per_entry), axis=0)
        ts_diff_padded[:front_pad_len] = pad_view.flat[-front_pad_len:]
        ts_diff_padded[-back_pad_len:] = pad_view.flat[:back_pad_len]

        # perform convolution
        smoothed_diff = ssignal.convolve(
            ts_diff_padded,
            kernel,
            mode="valid",
        ).squeeze()
        assert len(smoothed_diff) == len(ts_diff)  # sanity check sufficient padding

        # get strided values if desired
        # TODO: make more efficient? tons of unused computations in full convolve
        if stride_len > 1:
            smoothed_diff = smoothed_diff[(stride_len // 2) :: stride_len]
            smoothed_diff = np.repeat(smoothed_diff, stride_len)

        # build timestamps and align with original end
        smoothed_timestamps = np.empty(timestamps.shape, dtype=np.float128)
        # (assume first entry is same as second)
        smoothed_timestamps[:frames_per_entry] = np.cumsum(smoothed_diff[:frames_per_entry])
        smoothed_timestamps[frames_per_entry:] = np.cumsum(smoothed_diff) + smoothed_timestamps[frames_per_entry - 1]
        smoothed_timestamps += timestamps[-1] - smoothed_timestamps[-1]  # align timestamp end

        # optional: check that smoothed timestamps do not precede original
        # since that should not be possible
        if causal_check:
            assert np.all((timestamps - smoothed_timestamps) > 0.0)

        return smoothed_timestamps
