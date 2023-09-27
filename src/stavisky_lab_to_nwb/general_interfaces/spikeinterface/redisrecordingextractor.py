"""Redis recording extractor."""
import redis
import numpy as np
from typing import Union, Optional, Literal

from spikeinterface.core import BaseRecording, BaseRecordingSegment

from ...utils.redis import read_entry
from ...utils.timestamps import get_stream_ids_and_timestamps, smooth_timestamps


class RedisStreamRecordingExtractor(BaseRecording):
    """Recording extractor for recording data stored in Redis stream"""

    def __init__(
        self,
        port: int,
        host: str,
        stream_name: str,
        data_field: Union[bytes, str],
        data_dtype: Union[str, type, np.dtype],
        channel_ids: Optional[list] = None,
        frames_per_entry: int = 1,
        sampling_frequency: Optional[float] = None,
        timestamp_field: Optional[str] = None,
        timestamp_kwargs: dict = dict(),
        smoothing_kwargs: dict = dict(),
        gain_to_uv: Optional[float] = None,
        channel_dim: int = 0,
        chunk_size: int = 10000,
    ):
        """Initialize the RedisStreamRecordingExtractor

        Parameters
        ----------
        port : int
            Port number for Redis server
        host : str
            Host name for Redis server, e.g. "localhost"
        stream_name : str
            Name of stream containing the recording data
        data_key : bytes or str
            Key or field within each Redis stream entry that
            contains the recording data
        data_dtype : str, type, or numpy.dtype
            The dtype of the data. Assumed to be a numeric type
            recognized by numpy, e.g. int8, float32, etc.
        channel_ids : list, optional
            List of ids for each channel
        frames_per_entry : int, default: 1
            Number of frames (i.e. a single time point) contained
            within each Redis stream entry
        sampling_frequency : float, optional
            The sampling frequency of the data in Hz. See
            `RedisExtractorMixin`
        timestamp_source : bytes or str, optional
            The source of the timestamp information in the Redis
            stream. See `RedisExtractorMixin`
        timestamp_kwargs : dict, default: {}
            If timestamp source is not "redis", then timestamp_kwargs
            must contain the conversion factor, dtype, and encoding of
            the timestamp data, and optionally smoothing parameters. See
            `RedisExtractorMixin.get_ids_and_timestamps()`
        gain_to_uv : float, optional
            Scaling necessary to convert the recording values to
            microvolts
        channel_dim : int, default: 0
            If frames_per_entry > 1, then channel_dim indicates the
            axis ordering of the data in each entry. If channel_dim
            = 0, then the data is assumed to originally be of shape
            (channel_count, frames_per_entry). If channel_dim = 1,
            then the data is assumed to originally be of shape
            (frames_per_entry, channel_count)
        """
        # Instantiate Redis client and check connection
        self._client = redis.Redis(
            port=port,
            host=host,
        )
        self._client.ping()

        # check args and data validity
        stream_len = self._client.xlen(stream_name)
        assert stream_len > 0, "Stream has length 0"

        # get entry IDs and timestamps
        entry_ids, timestamps, nsp_timestamps = get_stream_ids_and_timestamps(
            client=self._client,
            stream_name=stream_name,
            frames_per_entry=frames_per_entry,
            timestamp_field=timestamp_field,
            chunk_size=chunk_size,
            **timestamp_kwargs,
        )
        if smoothing_kwargs:
            timestamps = smooth_timestamps(
                timestamps, 
                frames_per_entry=frames_per_entry,
                sampling_frequency=sampling_frequency, 
                **smoothing_kwargs
            )
        if sampling_frequency is None:
            sampling_frequency = np.round(1. / np.mean(np.diff(timestamps)), 8)

        # Construct channel IDs if not provided
        entry = self._client.xrange(stream_name, count=1)[0][1]
        data_size = read_entry(entry, data_field, dtype=data_dtype, encoding="buffer").size
        assert data_size % frames_per_entry == 0, "Size of Redis array must be multiple of frames_per_entry"
        channel_count = data_size // frames_per_entry
        if channel_ids is None:
            channel_ids = np.arange(channel_count, dtype=int).tolist()
        else:
            assert len(channel_ids) == channel_count, "Detected more channels than the number of channel IDS provided"
        
        # set up data reading args
        shape = (frames_per_entry, channel_count) if channel_dim == 1 else (channel_count, frames_per_entry)
        transpose = (channel_dim == 0)
        data_kwargs = dict(encoding="buffer", shape=shape, transpose=transpose)

        # Initialize Recording and RecordingSegment
        # NOTE: does not support multiple segments, assumes continuous recording for whole stream
        BaseRecording.__init__(self, channel_ids=channel_ids, sampling_frequency=sampling_frequency, dtype=data_dtype)
        recording_segment = RedisStreamRecordingSegment(
            client=self._client,
            stream_name=stream_name,
            data_field=data_field,
            data_dtype=data_dtype,
            data_kwargs=data_kwargs,
            timestamps=timestamps,
            entry_ids=entry_ids,
            nsp_timestamps=nsp_timestamps,
            frames_per_entry=frames_per_entry,
            t_start=None,
        )
        self.add_recording_segment(recording_segment)

        # Set gains if provided
        if gain_to_uv is not None:
            self.set_channel_gains(gain_to_uv)

        # Save some info in kwargs
        self._kwargs = {
            "port": port,
            "host": host,
            "stream_name": stream_name,
            "data_field": data_field,
            "data_dtype": data_dtype,
            "frames_per_entry": frames_per_entry,
        }

    def get_nsp_times(self, segment_index=None):
        segment_index = self._check_segment_index(segment_index)
        rs = self._recording_segments[segment_index]
        nsp_times = rs.get_nsp_times()
        return nsp_times

    def set_nsp_times(self, times, segment_index=None, with_warning=True):
        segment_index = self._check_segment_index(segment_index)
        rs = self._recording_segments[segment_index]

        assert times.ndim == 1, "Time must have ndim=1"
        assert rs.get_num_samples() == times.shape[0], "times have wrong shape"

        rs._nsp_timestamps = times.astype("float64")

        if with_warning:
            warn(
                "Setting times with Recording.set_nsp_times() is not recommended because "
                "times are not always propagated to across preprocessing"
                "Use use this carefully!"
            )
    
    def get_entry_ids(self):
        segment_index = self._check_segment_index(segment_index)
        rs = self._recording_segments[segment_index]
        entry_ids = rs.get_entry_ids()
        return entry_ids


class RedisStreamRecordingSegment(BaseRecordingSegment):
    def __init__(
        self,
        client: redis.Redis,
        stream_name: str,
        data_field: Union[bytes, str],
        data_dtype: Union[str, type, np.dtype],
        data_kwargs: dict,
        timestamps: np.ndarray,
        entry_ids: list[bytes],
        nsp_timestamps: Optional[np.ndarray] = None,
        frames_per_entry: int = 1,
        t_start: Optional[float] = None,
    ):
        """Initialize the RedisStreamRecordingSegment

        Parameters
        ----------
        client : redis.Redis
            Redis client connected to Redis server containing data
        stream_name : str
            Name of stream containing the recording data
        data_key : bytes or str
            Key or field within each Redis stream entry that
            contains the recording data
        data_dtype : str, type, or numpy.dtype
            The dtype of the data. Assumed to be a numeric type
            recognized by numpy, e.g. int8, float32, etc.
        channel_count : int
            Number of recording channels
        timestamps : numpy.ndarray
            An array containing timestamps for each frame
        entry_ids :  list of bytes
            A list containing the entry ID for each entry in the Redis
            stream in order
        frames_per_entry : int, default: 1
            Number of frames (i.e. a single time point) contained
            within each Redis stream entry
        t_start : float, optional
            The start time of the segment, relative to the recording
            start time
        channel_dim : int, default: 0
            If frames_per_entry > 1, then channel_dim indicates the
            axis ordering of the data in each entry. If channel_dim
            = 0, then the data is assumed to originally be of shape
            (channel_count, frames_per_entry). If channel_dim = 1,
            then the data is assumed to originally be of shape
            (frames_per_entry, channel_count)
        """
        # initialize base class
        BaseRecordingSegment.__init__(self, time_vector=timestamps, t_start=t_start)

        # Assign Redis client and check connection
        self._client = client
        self._client.ping()

        # arg checks
        assert len(entry_ids) == self._client.xlen(stream_name)

        # save some variables
        self._nsp_timestamps = nsp_timestamps
        self._stream_name = stream_name
        self._data_field = data_field
        self._data_kwargs = data_kwargs
        self._data_dtype = data_dtype
        self._entry_ids = entry_ids
        self._frames_per_entry = frames_per_entry
        self._num_samples = frames_per_entry * len(entry_ids)

    def get_num_samples(self) -> int:
        """Returns number of samples in the segment"""
        return self._num_samples
    
    def get_entry_ids(self):
        return self._entry_ids

    def get_traces(
        self,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
        channel_indices: Optional[list] = None,
    ) -> np.ndarray:
        """Gets specified recording traces from Redis

        Parameters
        ----------
        start_frame : int, optional
            The index of the frame to start reading data from
        end_frame : int, optional
            The index of the frame to stop reading data from,
            exclusive
        channel_indices : list, optional
            List of channel indices to retrieve data for

        Returns
        -------
        traces : numpy.ndarray
            A (frames, channels) array containing raw voltage
            trace data from Redis
        """
        # handle None args
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = self._num_samples

        # arg check (not allowing negative indices currently)
        assert start_frame >= 0 and start_frame < self._num_samples
        assert end_frame > 0 and end_frame <= self._num_samples

        # convert to entry number and within-entry idx
        start_entry_idx = start_frame // self._frames_per_entry
        start_frame_idx = start_frame % self._frames_per_entry
        end_entry_idx = (end_frame - 1) // self._frames_per_entry  # inclusive

        # read needed entries
        entries = self._client.xrange(
            self._stream_name,
            min=self._entry_ids[start_entry_idx],
            max=self._entry_ids[end_entry_idx],
        )

        # loop through, convert to numpy and stack
        traces = np.concatenate([
            read_entry(entry=entry[1], field=self._data_field, dtype=self._data_dtype, **self._data_kwargs) 
            for entry in entries
        ], axis=0)

        # slicing operations
        if channel_indices is not None:
            traces = traces[:, channel_indices]
        traces = traces[start_frame_idx : (start_frame_idx + end_frame - start_frame)]

        return traces

    def get_nsp_times(self):
        return self._nsp_timestamps