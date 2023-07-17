"""Redis recording extractor."""
import redis
import numpy as np
from typing import Union, Optional, Literal

from spikeinterface.core import BaseRecording, BaseRecordingSegment
from stavisky_lab_to_nwb.redis_interfaces.redisextractormixin import RedisExtractorMixin


class RedisStreamRecordingExtractor(BaseRecording, RedisExtractorMixin):
    """Recording extractor for recording data stored in Redis stream"""

    def __init__(
        self,
        port: int,
        host: str,
        stream_name: str,
        data_key: Union[bytes, str],
        dtype: Union[str, type, np.dtype],
        channel_ids: Optional[list] = None,
        frames_per_entry: int = 1,
        sampling_frequency: Optional[float] = None,
        timestamp_source: Optional[Union[bytes, str]] = None,
        timestamp_kwargs: dict = {},
        gain_to_uv: Optional[float] = None,
        channel_dim: int = 0,
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
        dtype : str, type, or numpy.dtype
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
        data_key = bytes(data_key, "utf-8") if isinstance(data_key, str) else data_key

        # get entry IDs and timestamps
        sampling_frequency, timestamps, entry_ids = self.get_ids_and_timestamps(
            stream_name=stream_name,
            frames_per_entry=frames_per_entry,
            sampling_frequency=sampling_frequency,
            timestamp_source=timestamp_source,
            **timestamp_kwargs,
        )

        # Construct channel IDs if not provided
        entry_data = self._client.xrange(stream_name, count=1)[0][1]
        data_size = np.frombuffer(entry_data[data_key], dtype=dtype).size
        assert data_size % frames_per_entry == 0, "Size of Redis array must be multiple of frames_per_entry"
        channel_count = data_size // frames_per_entry
        if channel_ids is None:
            channel_ids = np.arange(channel_count, dtype=int).tolist()
        else:
            assert len(channel_ids) == channel_count, "Detected more channels than the number of channel IDS provided"

        # Initialize Recording and RecordingSegment
        # NOTE: does not support multiple segments, assumes continuous recording for whole stream
        BaseRecording.__init__(self, channel_ids=channel_ids, sampling_frequency=sampling_frequency, dtype=dtype)
        recording_segment = RedisStreamRecordingSegment(
            client=self._client,
            stream_name=stream_name,
            data_key=data_key,
            dtype=dtype,
            channel_count=channel_count,
            timestamps=timestamps,
            entry_ids=entry_ids,
            frames_per_entry=frames_per_entry,
            t_start=None,
            channel_dim=channel_dim,
        )
        self.add_recording_segment(recording_segment)

        # Set gains if provided
        if gain_to_uv is not None:
            self.set_channel_gains(gain_to_uv)

        # Potential TODO: figure out what needs to be stored here
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


class RedisStreamRecordingSegment(BaseRecordingSegment):
    def __init__(
        self,
        client: redis.Redis,
        stream_name: str,
        data_key: Union[bytes, str],
        dtype: Union[str, type, np.dtype],
        channel_count: int,
        timestamps: np.ndarray,
        entry_ids: list[bytes],
        frames_per_entry: int = 1,
        t_start: Optional[float] = None,
        channel_dim: int = 0,  # TODO: confusing name?
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
        dtype : str, type, or numpy.dtype
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
        """Returns number of samples in the segment"""
        return self._num_samples

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
        stream_entries = self._client.xrange(
            self._stream_name,
            min=self._entry_ids[start_entry_idx],
            max=self._entry_ids[end_entry_idx],
        )

        # loop through, convert to numpy and stack
        traces = []
        for entry in stream_entries:
            entry_data = np.frombuffer(entry[1][self._data_key], dtype=self._dtype)
            assert entry_data.size == (self._frames_per_entry * self._channel_count)
            if self._frames_per_entry > 1:
                if self._channel_dim == 0:
                    entry_data = entry_data.reshape((self._channel_count, self._frames_per_entry)).T
                elif self._channel_dim == 1:
                    entry_data = entry_data.reshape((self._frames_per_entry, self._channel_count))
            else:
                entry_data = entry_data[None, :]
            if channel_indices is not None:
                entry_data = entry_data[:, channel_indices]
            traces.append(entry_data)
        traces = np.concatenate(traces, axis=0)

        # slicing operations
        traces = traces[start_frame_idx : (start_frame_idx + end_frame - start_frame)]

        return traces
