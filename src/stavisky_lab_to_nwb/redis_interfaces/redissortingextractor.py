"Redis sorting extractor."
import redis
import numpy as np
import pynwb
from typing import Union, Optional, List, Tuple, Literal

from spikeinterface.core import BaseSorting, BaseSortingSegment
from stavisky_lab_to_nwb.redis_interfaces.redisextractormixin import RedisExtractorMixin


class RedisStreamSortingExtractor(BaseSorting, RedisExtractorMixin):
    def __init__(
        self, 
        port: int,
        host: str,
        stream_name: str,
        data_key: Union[bytes, str],
        unit_count: int,
        dtype: Union[str, type, np.dtype],
        unit_ids: Optional[list] = None,
        frames_per_entry: int = 1,
        timestamps: Optional[list] = None,
        start_time: Optional[float] = None,
        sampling_frequency: Optional[float] = None,
        timestamp_source: Union[bytes, Literal["redis"]] = "redis",
        timestamp_kwargs: dict = {},
        unit_dim: int = 0,
    ):
        # Instantiate Redis client and check connection
        self._client = redis.Redis(
            port=port,
            host=host,
        )
        self._client.ping()
        
        # Construct unit IDs if not provided
        if unit_ids is None:
            unit_ids = np.arange(unit_count, dtype=int).tolist()
        
        # timestamp kwargs
        default_ts_kwargs = { # TODO: is this a good way to do things?
            "timestamp_unit": "ms",
            "chunk_size": 1000,
            "smoothing_window": 1000,
            "smoothing_stride": 1,
        }
        default_ts_kwargs.update(timestamp_kwargs)
        
        # get entry IDs and timestamps
        stream_len = self._client.xlen(stream_name)
        assert stream_len > 0, "Stream has length 0"
        start_time, sampling_frequency, timestamps, entry_ids = self.get_ids_and_timestamps(
            stream_name=stream_name,
            frames_per_entry=frames_per_entry,
            timestamps=timestamps,
            timestamp_source=timestamp_source,
            start_time=start_time,
            sampling_frequency=sampling_frequency,
            **default_ts_kwargs
        )
        
        # Initialize Sorting and SortingSegment
        # NOTE: does not support multiple segments, assumes continuous recording for whole stream
        BaseSorting.__init__(self, unit_ids=unit_ids, sampling_frequency=sampling_frequency)
        sorting_segment = RedisStreamSortingSegment(
            client=self._client,
            stream_name=stream_name,
            data_key=data_key,
            unit_count=unit_count,
            unit_ids=unit_ids,
            dtype=dtype,
            frames_per_entry=frames_per_entry,
            start_time=start_time,
            # sampling_frequency=sampling_frequency,
            timestamps=timestamps,
            entry_ids=entry_ids,
            unit_dim=unit_dim,
        )
        self.add_sorting_segment(sorting_segment)
        
        # Not sure what this is for?
        self._kwargs = {
            "port": port,
            "host": host,
            "stream_name": stream_name,
            "data_key": data_key,
            "unit_count": unit_count,
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
        BaseSortingSegment.__init__(self, t_start=start_time)
        
        # Assign Redis client and check connection
        self._client = client
        self._client.ping()
        
        # save some variables
        self._stream_name = stream_name
        self._data_key = bytes(data_key, "utf-8") if isinstance(data_key, str) else data_key
        self._unit_count = unit_count
        self._unit_ids = unit_ids
        self._unit_dim = unit_dim
        self._dtype = dtype
        self._entry_ids = entry_ids
        self._frames_per_entry = frames_per_entry
        self._num_samples = frames_per_entry * len(entry_ids)
        self._timestamps = timestamps
    
        # make chunk size an init arg?
        self._load_spike_trains(chunk_size=5000)
    
    def _load_spike_trains(self, chunk_size=5000):
        # load all spike trains
        stream_entries = self._client.xrange(self._stream_name, count=chunk_size)
        frame_counter = 0
        spike_frames = []
        spike_labels = []
        while len(stream_entries) > 0:
            for entry in stream_entries:
                entry_data = np.frombuffer(entry[1][self._data_key], dtype=self._dtype)
                assert entry_data.size == (self._frames_per_entry * self._unit_count)
                if self._frames_per_entry > 1:
                    if self._unit_dim == 0:
                        entry_data = entry_data.reshape((self._unit_count, self._frames_per_entry)).T
                    elif self._unit_dim == 1:
                        entry_data = entry_data.reshape((self._frames_per_entry, self._unit_count))
                else:
                    entry_data = entry_data[None, :]
                stime, scolumn = np.nonzero(entry_data)
                spike_times = np.repeat(stime, entry_data[stime, scolumn]) + frame_counter
                spike_units = np.repeat(scolumn, entry_data[stime, scolumn])
                spike_frames.append(spike_times)
                spike_labels.append(spike_units)
                
                frame_counter += self._frames_per_entry
            stream_entries = self._client.xrange(
                self._stream_name, min=b'(' + stream_entries[-1][0], count=chunk_size)
        spike_frames = np.concatenate(spike_frames, axis=0)
        spike_labels = np.concatenate(spike_labels, axis=0)
        self._spike_frames = [
            spike_frames[spike_labels == i] for i in range(self._unit_count)]
                
    def get_unit_spike_train(
        self,
        unit_id,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
    ) -> np.ndarray:
        # get unit idx
        unit_idx = self._unit_ids.index(unit_id)
        
        # handle None args
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = self._num_samples
        
        # arg check (not allowing negative indices currently)
        assert start_frame >= 0 and start_frame < self._num_samples
        assert end_frame > 0 and end_frame <= self._num_samples
        
        spike_frames = self._spike_frames[unit_idx]
        spike_frames = spike_frames[(spike_frames >= start_frame) & (spike_frames < end_frame)]
        
        return spike_frames
