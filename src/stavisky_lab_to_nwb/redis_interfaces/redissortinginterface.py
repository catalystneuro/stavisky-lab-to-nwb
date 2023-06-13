"Redis sorting extractor."
import redis
import numpy as np
from typing import Union, Optional, List, Tuple, Sequence, Literal

from spikeinterface.core import BaseSorting, BaseSortingSegment
from stavisky_lab_to_nwb.redis_interfaces.redisinterfacemixin import RedisInterfaceMixin


class RedisStreamSortingExtractor(BaseSorting, RedisInterfaceMixin):
    def __init__(
        self, 
        port: int,
        host: str,
        stream_name: str,
        data_key: Union[bytes, str],
        channel_count: int,
        dtype: Union[str, type, np.dtype],
        channel_ids: Optional[Sequence] = None,
        frames_per_entry: int = 1,
        timestamps: Optional[np.ndarray] = None,
        start_time: Optional[float] = None,
        sampling_frequency: Optional[float] = None,
        timestamp_source: Union[bytes, Literal["redis"]] = "redis",
        timestamp_kwargs: dict = {},
        gain_to_uv: Optional[float] = None,
        channel_dim: int = 0,
    ):
        # Instantiate Redis client and check connection
        self._client = redis.Redis(
            port=port,
            host=host,
        )
        self._client.ping()
        
        # Construct channel IDs if not provided
        if channel_ids is None:
            channel_ids = np.arange(channel_count, dtype=int).tolist()
        
        # timestamp kwargs
        default_ts_kwargs = { # TODO: is this a good way to do things?
            "timestamp_unit": "ms",
            "chunk_size": 1000,
            "smoothing_window": 30000,
            "smoothing_stride": 1,
        }
        default_ts_kwargs.update(timestamp_kwargs)
        
        # get entry IDs and timestamps
        stream_len = self._client.xlen(stream_name)
        assert stream_len > 0, "Stream has length 0"
        start_time, sampling_frequency, timestamps, entry_ids = self.get_ids_and_timestamps(
            stream_name=stream_name,
            frames_per_entry=frames_per_entry,
            timestamp_source=timestamp_source,
            start_time=start_time,
            sampling_frequency=sampling_frequency,
            **default_ts_kwargs
        )
        
        # Initialize Recording and RecordingSegment
        # NOTE: does not support multiple segments, assumes continuous recording for whole stream
        BaseRecording.__init__(self, channel_ids=channel_ids, sampling_frequency=sampling_frequency, dtype=dtype)
        recording_segment = RedisStreamRecordingSegment(
            client=self._client,
            stream_name=stream_name,
            data_key=data_key,
            channel_count=channel_count,
            dtype=dtype,
            frames_per_entry=frames_per_entry,
            start_time=start_time,
            # sampling_frequency=sampling_frequency,
            timestamps=timestamps,
            entry_ids=entry_ids,
            channel_dim=channel_dim,
        )
        self.add_recording_segment(recording_segment)
        
        # Set gains if provided
        if gain_to_uv is not None:
            self.set_channel_gains(gain_to_uv)
        
        # Not sure what this is for?
        self._kwargs = {
            "port": port,
            "host": host,
            "stream_name": stream_name,
            "data_key": data_key,
            "channel_count": channel_count,
            "dtype": dtype,
            "frames_per_entry": frames_per_entry,
            "timestamp_source": timestamp_source,
            # "timestamp_kwargs": timestamp_kwargs,
        }



class RedisStreamSortingSegment(BaseSortingSegment):
    def __init__(
        self, 
        client: redis.Redis,
        stream_name: str,
        data_key: Union[bytes, str],
        unit_ids: list,
        unit_count: int,
        dtype: Union[str, type, np.dtype],
        timestamps: np.ndarray,
        entry_ids: list[bytes],
        frames_per_entry: int = 1,
        start_time: Optional[float] = None,
        unit_dim: int = 0,
    ):
        BaseSegment.__init__(self, t_start=start_time)
        
        # Assign Redis client and check connection
        self._client = client
        self._client.ping()
        
        # save some variables
        self._stream_name = stream_name
        self._data_key = data_key
        self._unit_count = unit_count
        self._unit_dim = unit_dim
        self._dtype = dtype
        self._entry_ids = entry_ids
        self._frames_per_entry = frames_per_entry
        self._num_samples = frames_per_entry * len(entry_ids)
        self._timestamps = timestamps

    def get_unit_spike_train(
        self,
        unit_id,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
    ) -> np.ndarray:
        # get unit idx
        unit_idx = unit_ids.index(unit_id)
        
        # handle None args
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = self._num_samples - 1 # inclusive
        
        # arg check (not allowing negative indices currently)
        assert start_frame >= 0
        assert end_frame < self._num_samples
        
        # convert to entry number and within-entry idx
        start_entry_idx = start_frame // self._frames_per_entry
        start_frame_idx = start_frame % self._frames_per_entry
        end_entry_idx = end_frame // self._frames_per_entry
        end_frame_idx = end_frame % self._frames_per_entry
        
        # read needed entries
        stream_entries = self._client.xrange(
            self._stream_name,
            min=self._entry_ids[start_entry_idx],
            max=self._entry_ids[end_entry_idx],
        )
        
        # loop through, convert to numpy and stack
        spike_idx = []
        for n, entry in enumerate(stream_entries):
            entry_data = np.frombuffer(entry[1][self._data_key], dtype=self._dtype)
            assert entry_data.size == (self._frames_per_entry * self._unit_count)
            if self._frames_per_entry > 1:
                if self._unit_dim == 0:
                    entry_data = entry_data.reshape((self._unit_count, self._frames_per_entry)).T
                elif self._unit_dim == 1:
                    entry_data = entry_data.reshape((self._frames_per_entry, self._unit_count))
            entry_data = entry_data[:, unit_idx]
            if n == 0 and start_frame_idx > 0:
                entry_data = entry_data[start_frame_idx:]
            if n == len(stream_entries) - 1 and end_frame_idx < (self._frames_per_entry - 1):
                entry_data = entry_data[:(end_frame_idx + 1 - self._frames_per_entry)]
            has_spike = np.nonzero(entry_data)
            spike_times = (n + start_entry_idx) * frames_per_entry + np.repeat(has_spike, entry_data[has_spike]) + \
                float(n == 0) * start_frame_idx
            spike_idx.append(spike_times)
        spike_idx = np.concatenate(spike_idx, axis=0)
        
        spike_train = self._timestamps[spike_idx]
        
        return spike_train