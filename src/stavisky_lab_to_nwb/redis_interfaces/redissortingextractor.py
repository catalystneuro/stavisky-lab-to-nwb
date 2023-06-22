"Redis sorting extractor."
import redis
import numpy as np
import pynwb
from typing import Union, Optional, List, Tuple, Literal

from spikeinterface.core import BaseSorting, BaseSortingSegment, BaseRecording
from stavisky_lab_to_nwb.redis_interfaces.redisextractormixin import RedisExtractorMixin


class RedisStreamSortingExtractor(BaseSorting, RedisExtractorMixin):
    def __init__(
        self, 
        port: int,
        host: str,
        stream_name: str,
        data_key: Union[bytes, str],
        dtype: Union[str, type, np.dtype],
        unit_count: int,
        unit_ids: Optional[list] = None,
        frames_per_entry: int = 1,
        start_time: Optional[float] = None,
        sampling_frequency: Optional[float] = None,
        timestamp_source: Optional[Union[bytes, str]] = None,
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
        
        # get entry IDs and timestamps
        stream_len = self._client.xlen(stream_name)
        assert stream_len > 0, "Stream has length 0"
        start_time, sampling_frequency, timestamps, entry_ids = self.get_ids_and_timestamps(
            stream_name=stream_name,
            frames_per_entry=frames_per_entry,
            start_time=start_time,
            sampling_frequency=sampling_frequency,
            timestamp_source=timestamp_source,
            **timestamp_kwargs
        )
        
        # Initialize Sorting and SortingSegment
        # NOTE: does not support multiple segments, assumes continuous recording for whole stream
        BaseSorting.__init__(self, unit_ids=unit_ids, sampling_frequency=sampling_frequency)
        sorting_segment = RedisStreamSortingSegment(
            client=self._client,
            stream_name=stream_name,
            data_key=data_key,
            dtype=dtype,
            unit_ids=unit_ids,
            entry_ids=entry_ids,
            frames_per_entry=frames_per_entry,
            t_start=0, # t_start != start_time
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
    
    def register_recording(
        self, 
        recording: BaseRecording, 
        check_spike_frames: bool = True,
        frequency_ratio: int = 1,
        override_frequency: bool = False,
    ):
        # check that recording lengths match up
        sorting_samples = 0
        for segment_index in range(self.get_num_segments()):
            sorting_samples += self._sorting_segments[segment_index].get_num_samples()
        assert (sorting_samples * frequency_ratio) == recording.get_num_samples()
        # give a little more leeway in frequency mismatch between sorting and recording
        if override_frequency:
            if np.isclose( # completely arbitrary rtol of 1e-3
                self.get_sampling_frequency() * frequency_ratio, recording.get_sampling_frequency(), rtol=1e-3,
            ):
                self._sampling_frequency = recording.get_sampling_frequency() # this is probably a terrible idea
            else:
                import pdb; pdb.set_trace()
                raise AssertionError("The recording has a different sampling frequency than the sorting!")
        else:
            self._sampling_frequency *= frequency_ratio
        for segment in self._sorting_segments:
            segment.set_frame_scaling(frequency_ratio)
        super().register_recording(recording) #, check_spike_frames=check_spike_frames)
        
        
class RedisStreamSortingSegment(BaseSortingSegment):
    def __init__(
        self, 
        client: redis.Redis,
        stream_name: str,
        data_key: Union[bytes, str],
        dtype: Union[str, type, np.dtype],
        unit_ids: list,
        entry_ids: list[bytes],
        frames_per_entry: int = 1,
        t_start: Optional[float] = None,
        unit_dim: int = 0,
    ):
        # initialize base class
        BaseSortingSegment.__init__(self, t_start=t_start)
        
        # assign Redis client and check connection
        self._client = client
        self._client.ping()
        
        # arg checks
        assert unit_dim in [0, 1]
        assert len(entry_ids) == self._client.xlen(stream_name)
        
        # save some variables
        self._stream_name = stream_name
        self._data_key = bytes(data_key, "utf-8") if isinstance(data_key, str) else data_key
        self._dtype = dtype
        self._unit_count = len(unit_ids)
        self._unit_ids = unit_ids
        self._unit_dim = unit_dim
        self._entry_ids = entry_ids
        self._frames_per_entry = frames_per_entry
        self._num_samples = frames_per_entry * len(entry_ids)
        self._frame_scaling = 1
    
        # make chunk size an init arg?
        self._spike_frames = None
        self._load_spike_frames(chunk_size=1000)
    
    def _load_spike_frames(self, chunk_size: int = 1000):
        # initialize loop variables
        stream_entries = self._client.xrange(self._stream_name, count=chunk_size)
        frame_counter = 0
        spike_frames = []
        spike_labels = []
        
        # loop until all entries read
        while len(stream_entries) > 0:
            for entry in stream_entries:
                # read data and check expected size
                entry_data = np.frombuffer(entry[1][self._data_key], dtype=self._dtype)
                assert entry_data.size == (self._frames_per_entry * self._unit_count)
                # reshape array to 2d
                if self._frames_per_entry > 1:
                    if self._unit_dim == 0:
                        entry_data = entry_data.reshape((self._unit_count, self._frames_per_entry)).T
                    elif self._unit_dim == 1:
                        entry_data = entry_data.reshape((self._frames_per_entry, self._unit_count))
                else:
                    entry_data = entry_data[None, :]
                # find nonzero entries
                stime, scolumn = np.nonzero(entry_data)
                # if spike count > 1, repeat that frame
                entry_spike_frames = np.repeat(stime, entry_data[stime, scolumn]) + frame_counter
                entry_spike_labels = np.repeat(scolumn, entry_data[stime, scolumn])
                # append to list
                spike_frames.append(entry_spike_frames)
                spike_labels.append(entry_spike_labels)
                # update base frame number
                frame_counter += self._frames_per_entry
            # load next chunk of entries
            stream_entries = self._client.xrange(
                self._stream_name, min=b'(' + stream_entries[-1][0], count=chunk_size)
        # stack all spike frames and labels
        spike_frames = np.concatenate(spike_frames, axis=0)
        spike_labels = np.concatenate(spike_labels, axis=0)
        # double check outputs
        assert len(spike_frames) == len(spike_labels)
        assert np.all((spike_labels >= 0) & (spike_labels < self._unit_count))
        # create list of views into spike frames indexed by label/unit
        self._spike_frames = [
            spike_frames[spike_labels == i] for i in range(self._unit_count)]
    
    def set_frame_scaling(self, frame_scaling: int = 1):
        # useful if sorting data is already binned to lower frequency than
        # recording, but we want timestamps to be shared
        self._frame_scaling = frame_scaling
    
    def get_num_samples(self) -> int:
        return self._num_samples

    def get_unit_spike_train(
        self,
        unit_id,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
    ) -> np.ndarray:
        # make sure data are loaded already
        assert self._spike_frames is not None
        
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
        
        # get spike frames and filter based on frame range
        spike_frames = self._spike_frames[unit_idx]
        spike_frames = spike_frames[(spike_frames >= start_frame) & (spike_frames < end_frame)]
        spike_frames *= self._frame_scaling
        
        return spike_frames
