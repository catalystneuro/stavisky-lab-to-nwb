import sys
import redis
import numpy as np
from typing import Union, Literal, Optional

from .redis_io import safe_decode, read_stream_fields


def get_stream_ids_and_timestamps(
    client: redis.Redis,
    stream_name: str,
    frames_per_entry: int = 1,
    timestamp_field: Optional[str] = None,
    timestamp_conversion: float = 1.0,
    timestamp_encoding: Optional[Literal["string", "buffer"]] = None,
    timestamp_dtype: Optional[Union[str, type, np.dtype]] = None,
    timestamp_index: Optional[int] = None,
    timestamp_shape: Optional[tuple] = None,
    buffer_gb: Optional[float] = None,
):
    """Reads a stream's entry IDs, converts those to timestamps,
    and optionally also reads another timestamp field."""
    # set up args for read_stream_fields
    if timestamp_index is not None:
        assert frames_per_entry == 1
        if timestamp_shape is None:
            timestamp_shape = (-1,)
    timestamp_kwargs = {}
    if timestamp_field is not None:
        assert timestamp_encoding is not None
        timestamp_kwargs.update(
            {
                timestamp_field: dict(
                    encoding=timestamp_encoding,
                    dtype=timestamp_dtype,
                    shape=timestamp_shape or (frames_per_entry,),
                    index=timestamp_index,
                )
            }
        )
    # read ids and timestamps
    field_data = read_stream_fields(
        client=client,
        stream_name=stream_name,
        field_kwargs=timestamp_kwargs,
        return_ids=True,
        buffer_gb=buffer_gb,
    )
    # convert entry ids to redis timestamps
    entry_ids = field_data["ids"]
    redis_timestamps = np.array([int(safe_decode(eid).split("-")[0]) * 1e-3 for eid in entry_ids], dtype=np.float128)
    redis_timestamps = np.repeat(redis_timestamps, frames_per_entry)
    field_timestamps = field_data.get(timestamp_field)
    if field_timestamps is not None:
        if isinstance(field_timestamps[0], np.ndarray):
            field_timestamps = np.concatenate(field_timestamps, axis=0).astype("float128")
        else:
            field_timestamps = np.array(field_timestamps).astype("float128")
        field_timestamps *= timestamp_conversion
        assert redis_timestamps.shape == field_timestamps.shape
    return entry_ids, redis_timestamps, field_timestamps


def interpolate_timestamps(
    timestamps: np.ndarray,
    frames_per_entry: int,
    sampling_frequency: Optional[float] = None,
):
    if sampling_frequency is None:
        sampling_diff = (timestamps[-1] - timestamps[frames_per_entry - 1]) / (len(timestamps) - frames_per_entry)
    else:
        sampling_diff = 1.0 / sampling_frequency
    smoothed_timestamps = np.arange(len(timestamps)) * sampling_diff
    return smoothed_timestamps


def convolve_smooth_timestamps(
    timestamps: np.ndarray,
    frames_per_entry: int,
    window_len: int = 1,
    stride_len: int = 1,
):
    # no smoothing if window_len of 1
    if window_len == 1:
        return timestamps
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

    return smoothed_timestamps


def smooth_timestamps(
    timestamps: np.ndarray,
    frames_per_entry: int,
    sampling_frequency: Optional[float] = None,
    window_len: Union[int, Literal["max"]] = 1,
    stride_len: int = 1,
    enforce_causal: bool = False,
):
    if window_len == 1:
        return timestamps
    elif window_len == "max":
        smoothed_timestamps = interpolate_timestamps(timestamps, frames_per_entry, sampling_frequency)
    else:
        smoothed_timestamps = convolve_smooth_timestamps(
            timestamps=timestamps,
            frames_per_entry=frames_per_entry,
            window_len=window_len,
            stride_len=stride_len,
        )

    smoothed_timestamps += timestamps[-1] - smoothed_timestamps[-1]  # align timestamp end

    # optional: check that smoothed timestamps do not precede original
    # since that should not be possible
    if enforce_causal:
        offset = np.min(timestamps - smoothed_timestamps)
        smoothed_timestamps += offset
    else:
        if np.any(timestamps < smoothed_timestamps):
            print("Warning: some smoothed timestamps occur after the original timestamps.")

    return smoothed_timestamps