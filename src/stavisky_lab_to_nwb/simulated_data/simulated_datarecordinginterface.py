"""Primary class for converting ecephys recording."""
from pynwb.file import NWBFile
from typing import Union, Optional, List, Tuple, Sequence

from neuroconv.basedatainterface import BaseDataInterface

from spikeinterface.extractors import BaseRecording, BaseRecordingSegment


class RedisStreamRecordingSegment(BaseRecordingSegment):
    def __init__(
        self, 
        client: redis.Redis,
        stream_name: str,
        data_key: Union[bytes, str],
        channel_count: int,
        dtype: Union[str, type, np.dtype],
        segment_start: int,
        segment_end: int,
        channel_ids: Optional[Sequence] = None,
        frames_per_entry: int = 1,
        start_time: Optional[float] = None,
        sampling_frequency: Optional[float] = None,
        timestamp_source: Union[bytes, str] = "redis",
        timestamp_kwargs: dict = {},
    ):
        # Assign Redis client and check connection
        self._client = client
        self._client.ping()
        
    def get_traces(
        self,
        start_frame: Union[int, None] = None,
        end_frame: Union[int, None] = None,
        channel_indices: Union[List, None] = None,
    ) -> np.ndarray:
        pass
        


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
    ):
        # Instantiate Redis client and check connection
        self._client = redis.Redis(
            port=port,
            host=host,
        )
        self._client.ping()
        
        # Construct channel IDs if not provided
        if channel_ids is None:
            channel_ids = np.arange(channel_ids).tolist()
        
        # Infer sampling frequency from first and last entry
        stream_len = self._client.xlen(stream_name)
        assert stream_len > 0, "Stream has length 0"
        if sampling_frequency is None:
            stream_start = int(self._client.xrange(stream_name, count=1)[0][0].split(b'-'))
            stream_end = int(self._client.xrevrange(stream_name, count=1)[0][0].split(b'-'))
            stream_dur = (stream_end - stream_start) / 1000.
            sampling_frequency = round((stream_len - 1) * frames_per_entry / stream_dur, 9)
        
        # Infer start time from sampling frequency and first entry
        if start_time is None:
            stream_start = stream_start = int(self._client.xrange(stream_name, count=1)[0][0].split(b'-'))
            start_time = stream_start / 1000. - frames_per_entry * sampling_frequency
        
        # Initialize Recording and RecordingSegment
        # NOTE: does not support multiple segments, assumes continuous recording for whole sream
        BaseRecording.__init__(self, channel_ids=channel_ids, sampling_frequency=sampling_frequency, dtype=dtype)
        recording_segment = RedisStreamRecordingSegment(
            client=self._client,
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


class RedisRecordingInterface(BaseDataInterface):
    """Recording interface for Stavisky Redis conversion"""

    def __init__(self):
        # This should load the data lazily and prepare variables you need
        pass

    def get_metadata(self):
        # Automatically retrieve as much metadata as possible
        metadata = super().get_metadata()

        return metadata

    def run_conversion(self, nwbfile: NWBFile, metadata: dict):
        # All the custom code to write to PyNWB

        return nwbfile