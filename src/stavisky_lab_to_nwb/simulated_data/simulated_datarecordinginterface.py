"""Primary class for converting ecephys recording."""
import redis
import numpy as np
from pynwb.file import NWBFile
from typing import Union, Optional, List, Tuple, Sequence, Literal

from neuroconv.datainterfaces.ecephys.baserecordingextractorinterface import BaseRecordingExtractorInterface

from spikeinterface.core import BaseRecording, BaseRecordingSegment


class RedisStreamRecordingSegment(BaseRecordingSegment):
    def __init__(
        self, 
        client: redis.Redis,
        stream_name: str,
        data_key: Union[bytes, str],
        channel_count: int,
        dtype: Union[str, type, np.dtype],
        timestamps: np.ndarray,
        entry_ids: list[bytes],
        frames_per_entry: int = 1,
        start_time: Optional[float] = None,
        sampling_frequency: Optional[float] = None,
        channel_dim: int = 0, # TODO: confusing name?
    ):
        # Assign Redis client and check connection
        self._client = client
        self._client.ping()
        
        # initialize base class
        BaseRecordingSegment.__init__(self, time_vector=timestamps, t_start=start_time)
        
        # arg checks
        assert channel_dim in [0, 1]
        assert len(entry_ids) == self._client.xlen(stream_name)
        
        # save some variables
        self._stream_name = stream_name
        self._data_key = data_key
        self._channel_count = channel_count
        self._channel_dim = channel_dim
        self._dtype = dtype
        self._entry_ids = entry_ids
        self._frames_per_entry = frames_per_entry
        self._num_samples = frames_per_entry * len(entry_ids)

    def get_num_samples(self) -> int:
        return self._num_samples
        
    def get_traces(
        self,
        start_frame: Union[int, None] = None,
        end_frame: Union[int, None] = None,
        channel_indices: Union[List, None] = None,
    ) -> np.ndarray:
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
        
        stream_entries = self._client.xrange(
            self._stream_name,
            min=self._entry_ids[start_entry_idx],
            max=self._entry_ids[end_entry_idx],
        )
        
        traces = []
        for entry in stream_entries:
            entry_data = np.frombuffer(entry[1][self._data_key], dtype=self._dtype)
            assert entry_data.size == (self._frames_per_entry * self._channel_count)
            if self._frames_per_entry > 1:
                if self._channel_dim == 0:
                    entry_data = entry_data.reshape((self._channel_count, self._frames_per_entry)).T
                elif self._channel_dim == 1:
                    entry_data = entry_data.reshape((self._frames_per_entry, self._channel_count))
            traces.append(entry_data)
        traces = np.concatenate(traces, axis=0)
        
        # slicing operations
        # TODO: make more compact
        if start_frame_idx > 0:
            traces = traces[start_frame_idx:]
        if end_frame_idx < self._frames_per_entry - 1:
            traces = traces[:(end_frame_idx + 1 - self._frames_per_entry)]
        if channel_indices is not None:
            traces = traces[:, channel_indices]
        
        return traces
        

class RedisStreamRecordingExtractor(BaseRecording):
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
        start_time: Optional[float] = None,
        sampling_frequency: Optional[float] = None,
        timestamp_source: Union[bytes, str] = "redis",
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
            channel_ids = np.arange(channel_count).tolist()
        
        # timestamp kwargs
        default_ts_kwargs = { # TODO: or just move these to init args?
            "timestamp_unit": "ms",
            "chunk_size": 100,
            "smooth_timestamps": True,
        }
        default_ts_kwargs.update(timestamp_kwargs)
        
        # Infer sampling frequency from first and last entry
        stream_len = self._client.xlen(stream_name)
        assert stream_len > 0, "Stream has length 0"
        start_time, sampling_frequency, timestamps, entry_ids = self.calculate_timestamps(
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
            sampling_frequency=sampling_frequency,
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
        
    def calculate_timestamps(
        self,
        stream_name: str,
        frames_per_entry: int,
        timestamp_source: Union[bytes, str],
        timestamp_unit: Literal["s", "ms", "us"],
        timestamp_encoding: Optional[Literal["str", "buffer"]] = None,
        timestamp_dtype: Optional[Union[str, type, np.dtype]] = None,
        start_time: Optional[float] = None,
        sampling_frequency: Optional[float] = None,
        chunk_size: int = 10,
        smooth_timestamps: bool = True,
    ):
        # NOTE: if this is useful for other classes, maybe it can be in a mixin
        
        # initialize variables for loop
        timestamps = []
        entry_ids = []
        last_ts = -np.inf
        
        # get first block of entries
        num_entries = self._client.xlen(stream_name)
        stream_entries = self._client.xrange(stream_name, count=chunk_size)
        
        # loop until all entries fetched
        while len(stream_entries) > 0:
            # store entry ids for indexing
            entry_ids += [entry[0] for entry in stream_entries]
            
            # build timestamps as specified
            for entry in stream_entries:
                # use redis stream ids as timestamps
                if timestamp_source.lower() == "redis":
                    # duplicate stream id as timestamp for each frame in entry
                    entry_timestamps = np.full((frames_per_entry,), float(entry[0].split(b'-')[0]))
                    assert entry_timestamps[0] >= last_ts # sanity check for monotonic increasing ts
                    last_ts = entry_timestamps[0]
                else:
                    assert timestamp_source in entry[1].keys() # not all entries have to have the same keys
                    if timestamp_encoding == "str":
                        # pretty risky casting string to dtype, but some timestamps are stored as strings
                        entry_timestamps = np.full((frames_per_entry,), timestamp_dtype(str(entry[1][timestamp_source])))
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
                # add timestamps to list
                timestamps.append(entry_timestamps)
            
            # read next entries
            last_id = stream_entries[-1][0]
            sub_ms_identifier = int(last_id.split(b'-')[-1])
            next_id = last_id.replace(
                b'-' + bytes(str(sub_ms_identifier), 'UTF-8'),
                b'-' + bytes(str(sub_ms_identifier + 1), 'UTF-8')
            )
            stream_entries = self._client.xrange(stream_name, min=next_id, count=chunk_size)
        
        # sanity check lengths
        assert len(timestamps) == num_entries
        assert len(entry_ids) == num_entries
        
        # join all timestamps
        timestamps = np.concatenate(timestamps)
        if timestamp_unit == "ms":
            timestamps *= 1e-3
        elif timestamp_unit == "us":
            timestamps *= 1e-6
        
        # compute frequency and period if necessary
        if sampling_frequency is None:
            sampling_period = float(timestamps[-1] - timestamps[frames_per_entry - 1]) / ((num_entries - 1.) * frames_per_entry)
            sampling_frequency = 1. / sampling_period
        else:
            sampling_period = 1. / sampling_frequency
        
        # "smooth" timestamps to regular
        if smooth_timestamps:
            # estimate starting timestamp by working backwards from end
            # (this is probably better when data come in bursts - the first sample has more delay
            # from its true timestamp than the last sample in a burst)
            stream_start_time = timestamps[-1] - (num_entries * frames_per_entry - 1) * sampling_period
            timestamps_smth = np.linspace(stream_start_time, timestamps[-1], num_entries * frames_per_entry)
            # subtract so that timestamps_smth <= timestamps, since the true timestamps can't
            # possibly precede the logged timestamps
            # diff = np.max(timestamps_smth - timestamps)
            # timestamps_smth -= diff
            timestamps = timestamps_smth
        
        # get start time if necessary
        if start_time is None:
            start_time = timestamps[0]
        timestamps -= start_time
        assert np.all(timestamps >= 0.) # sanity check
        
        return start_time, sampling_frequency, timestamps, entry_ids


class RedisRecordingInterface(BaseRecordingExtractorInterface):
    """Recording interface for Stavisky Redis conversion"""
    
    ExtractorModule = "stavisky_lab_to_nwb.simulated_data.simulated_datarecordinginterface"
    ExtractorName = "RedisStreamRecordingExtractor"

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
        start_time: Optional[float] = None,
        sampling_frequency: Optional[float] = None,
        timestamp_source: Union[bytes, str] = "redis",
        timestamp_kwargs: dict = {},
        gain_to_uv: Optional[float] = None,
        channel_dim: int = 0,
        verbose: bool = True,
        es_key: str = "ElectricalSeries",
    ):
        super().__init__(
            verbose=verbose,
            es_key=es_key,
            port=port,
            host=host,
            stream_name=stream_name,
            data_key=data_key,
            channel_count=channel_count,
            dtype=dtype,
            channel_ids=channel_ids,
            frames_per_entry=frames_per_entry,
            start_time=start_time,
            sampling_frequency=sampling_frequency,
            timestamp_source=timestamp_source,
            timestamp_kwargs=timestamp_kwargs,
            gain_to_uv=gain_to_uv,
            channel_dim=channel_dim,
        )

"""
if __name__ == "__main__":
    extractor = RedisStreamRecordingExtractor(
        port=6379,
        host="localhost",
        stream_name="continuousNeural",
        data_key=b'samples',
        channel_count=256,
        dtype=np.int16,
        channel_ids=None,
        frames_per_entry=30,
        start_time=None,
        sampling_frequency=None,
        timestamp_source="redis",
        timestamp_kwargs={
            "chunk_size": 1000,
        },
        gain_to_uv=-100.,
        channel_dim=1,
    )
    traces = extractor.get_traces(
        segment_index=None,
        start_frame=0,
        end_frame=30000,
        channel_ids=None,
        order=None,
        return_scaled=False,
        cast_unsigned=False,
    )
    import pdb; pdb.set_trace()
"""