"Redis sorting extractor."
import redis
import numpy as np
import pynwb
from typing import Union, Optional, List, Tuple, Literal
from warnings import warn

from spikeinterface.core import BaseSorting, BaseSortingSegment, BaseRecording
from ...utils.redis_io import read_entry
from ...utils.timestamps import get_stream_ids_and_timestamps


class RedisStreamSortingExtractor(BaseSorting):
    def __init__(
        self,
        port: int,
        host: str,
        stream_name: str,
        data_field: Union[bytes, str],
        data_dtype: Union[str, type, np.dtype],
        unit_ids: Optional[list] = None,
        frames_per_entry: int = 1,
        sampling_frequency: Optional[float] = None,
        nsp_timestamp_field: Optional[str] = None,
        nsp_timestamp_kwargs: dict = dict(),
        smoothing_kwargs: dict = dict(),
        unit_dim: int = 0,
        clock: Literal["redis", "nsp"] = "redis",
        chunk_size: Optional[int] = None,
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
        data_field : bytes or str
            Key or field within each Redis stream entry that
            contains the recording data
        data_dtype : str, type, or numpy.dtype
            The dtype of the data. Assumed to be a numeric type
            recognized by numpy, e.g. int8, float32, etc.
        unit_ids : list, optional
            List of ids for each unit
        frames_per_entry : int, default: 1
            Number of frames (i.e. a single time point) contained
            within each Redis stream entry
        sampling_frequency : float, optional
            The sampling frequency of the data in Hz. See
            `RedisExtractorMixin`
        nsp_timestamp_field : bytes or str, optional
            The field name of additional timestamps in the Redis
            stream entries. Redis timestamps will always be loaded
            and saved, but if `nsp_timestamp_field` is provided,
            the additional timestamps can be saved as well.
        nsp_timestamp_kwargs : dict, default: {}
            Necessary kwargs for reading the additional timestamps.
            See `get_stream_ids_and_timestamps()`.
        smoothing_kwargs : dict, default: {}
            Parameters for `utils.timestamps.smooth_timestamps()`. 
            Timestamp smoothing is currently only applied to the Redis
            timestamps
        unit_dim : int, default: 0
            If frames_per_entry > 1, then unit_dim indicates the
            axis ordering of the data in each entry. If unit_dim
            = 0, then the data is assumed to originally be of shape
            (unit_count, frames_per_entry). If unit_dim = 1,
            then the data is assumed to originally be of shape
            (frames_per_entry, unit_count)
        unit_dim : int, default: 0
            If frames_per_entry > 1, then unit_dim indicates the
            axis ordering of the data in each entry. If unit_dim
            = 0, then the data are assumed to originally be of shape
            (unit_count, frames_per_entry). If channel_dim = 1,
            then the data are assumed to originally be of shape
            (frames_per_entry, unit_count)
        clock : {"redis", "nsp"}, default: "redis"
            Which clock to use for spike times. If "redis",
            the real-time Redis timestamps will be used. If
            "nsp", the NSP timestamps will be used. The
            clock can be toggled after initialization
            with `RedisStreamSortingExtractor.set_clock()`
        chunk_size : int, optional
            The number of entries to read simultaneously when
            iterating through the Redis stream. If None,
            all entries will be read at once
        """
        # Instantiate Redis client and check connection
        self._client = redis.Redis(
            port=port,
            host=host,
        )
        self._client.ping()
        self._clock = clock

        # check args and data validity
        stream_len = self._client.xlen(stream_name)
        assert stream_len > 0, "Stream has length 0"

        # get entry IDs and timestamps
        entry_ids, timestamps, nsp_timestamps = get_stream_ids_and_timestamps(
            client=self._client,
            stream_name=stream_name,
            frames_per_entry=frames_per_entry,
            timestamp_field=nsp_timestamp_field,
            chunk_size=chunk_size,
            **nsp_timestamp_kwargs,
        )
        if smoothing_kwargs:
            timestamps = smooth_timestamps(
                timestamps, frames_per_entry=frames_per_entry, sampling_frequency=sampling_frequency, **smoothing_kwargs
            )
        if sampling_frequency is None:
            sampling_frequency = np.round(1.0 / np.mean(np.diff(timestamps)), 8)

        # Construct unit IDs if not provided
        entry = self._client.xrange(stream_name, count=1)[0][1]
        data_size = read_entry(entry, data_field, dtype=data_dtype, encoding="buffer").size
        assert data_size % frames_per_entry == 0, "Size of Redis array must be multiple of frames_per_entry"
        unit_count = data_size // frames_per_entry
        if unit_ids is None:
            unit_ids = np.arange(unit_count, dtype=int).tolist()
        else:
            assert len(unit_ids) == unit_count, "Detected more units than the number of unit IDs provided"

        # set up data reading args
        shape = (frames_per_entry, unit_count) if unit_dim == 1 else (unit_count, frames_per_entry)
        transpose = unit_dim == 0
        data_kwargs = dict(encoding="buffer", shape=shape, transpose=transpose)

        # Initialize Sorting and SortingSegment
        # NOTE: does not support multiple segments, assumes continuous recording for whole stream
        BaseSorting.__init__(self, unit_ids=unit_ids, sampling_frequency=float(sampling_frequency))
        sorting_segment = RedisStreamSortingSegment(
            client=self._client,
            stream_name=stream_name,
            data_field=data_field,
            data_dtype=data_dtype,
            data_kwargs=data_kwargs,
            unit_ids=unit_ids,
            entry_ids=entry_ids,
            timestamps=timestamps,
            nsp_timestamps=nsp_timestamps,
            frames_per_entry=frames_per_entry,
            t_start=None,
            chunk_size=chunk_size,
        )
        self.add_sorting_segment(sorting_segment)

        # Potential TODO: figure out what needs to be stored here
        self._kwargs = {
            "port": port,
            "host": host,
            "stream_name": stream_name,
            "data_field": data_field,
            "data_dtype": data_dtype,
            "frames_per_entry": frames_per_entry,
        }

    def get_unit_spike_train(
        self,
        unit_id,
        segment_index: Union[int, None] = None,
        start_frame: Union[int, None] = None,
        end_frame: Union[int, None] = None,
        return_times: bool = False,
        clock: Optional[Literal["redis", "nsp"]] = None,
    ):
        """Get spike train for a particular unit.
        Overriding base class method so that timestamps can
        also come from time vector instead of only start
        time and sampling frequency.

        Parameters
        ----------
        unit_id : int or str
            ID of the unit to retrieve
        segment_index : int, optional
            Index if the segment to get spike trains from. If
            there is only one segment, segment_index=None returns
            that segment
        start_frame : int, optional
            The frame to start fetching data from, if only a
            portion of the segment is to be returned
        end_frame : int, optional
            The frame to stop fetching data from (exclusive),
            if only a portion of the segment is to be returned
        return_times : bool, default: False
            Whether to return the frame indices of observed spikes
            (False) or to return their corresponding timestamps
            (True)
        """
        override_times = return_times and not self.has_recording()
        spike_frames = super().get_unit_spike_train(
            unit_id=unit_id,
            segment_index=segment_index,
            start_frame=start_frame,
            end_frame=end_frame,
            return_times=(return_times and not override_times),
        )
        if not override_times:
            return spike_frames
        else:
            segment = self._sorting_segments[segment_index]
            clock = clock or self._clock
            if clock == "redis":
                times = segment.get_times().astype("float64")
            else:
                times = segment.get_nsp_times().astype("float64")
            return times[spike_frames]

    def get_times(self, segment_index=None):
        """Get time vector for a sorting segment.

        If the segment has a time_vector, then it is returned. Otherwise
        a time_vector is constructed on the fly with sampling frequency.
        If t_start is defined and the time vector is constructed on the fly,
        the first time will be t_start. Otherwise it will start from 0.

        Parameters
        ----------
        segment_index : int, optional
            The segment index (required for multi-segment), by default None

        Returns
        -------
        np.array
            The 1d times array
        """
        if self.has_recording():
            return self._recording.get_times(segment_index=segment_index)
        segment_index = self._check_segment_index(segment_index)
        segment = self._sorting_segments[segment_index]
        times = segment.get_times()
        return times

    def set_times(self, times, segment_index=None, with_warning=True):
        """Set times for a sorting segment.

        Parameters
        ----------
        times : 1d np.array
            The time vector
        segment_index : int, optional
            The segment index (required for multi-segment), by default None
        with_warning : bool, optional
            If True, a warning is printed, by default True
        """
        if self.has_recording():
            self._recording.set_times(times=times, segment_index=segment_index, with_warning=with_warning)
        else:
            segment_index = self._check_segment_index(segment_index)
            segment = self._sorting_segments[segment_index]

            assert times.ndim == 1, "Time must have ndim=1"
            assert segment.get_num_samples() == times.shape[0], "times have wrong shape"

            segment.t_start = None
            segment.time_vector = times.astype("float64")

            if with_warning:
                warn(
                    "Setting times with Sorting.set_times() is not recommended because "
                    "times are not always propagated to across preprocessing"
                    "Use use this carefully!"
                )

    def get_nsp_times(self, segment_index=None):
        if self.has_recording():
            return self._recording.get_nsp_times(segment_index=segment_index)
        segment_index = self._check_segment_index(segment_index)
        segment = self._sorting_segments[segment_index]
        times = segment.get_nsp_times()
        return times

    def set_nsp_times(self, times, segment_index=None, with_warning=True):
        if self.has_recording():
            self._recording.set_nsp_times(times=times, segment_index=segment_index, with_warning=with_warning)
        else:
            segment_index = self._check_segment_index(segment_index)
            rs = self._sorting_segments[segment_index]

            assert times.ndim == 1, "Time must have ndim=1"
            assert rs.get_num_samples() == times.shape[0], "times have wrong shape"

            rs._nsp_timestamps = times.astype("float64")

            if with_warning:
                warn(
                    "Setting times with Recording.set_nsp_times() is not recommended because "
                    "times are not always propagated to across preprocessing"
                    "Use use this carefully!"
                )

    def get_entry_ids(self, segment_index=None):
        segment_index = self._check_segment_index(segment_index)
        segment = self._sorting_segments[segment_index]
        entry_ids = segment.get_entry_ids()
        return entry_ids

    def get_clock(self):
        return self._clock

    def set_clock(self, clock):
        assert clock in ["redis", "nsp"]
        self._clock = clock


class RedisStreamSortingSegment(BaseSortingSegment):
    def __init__(
        self,
        client: redis.Redis,
        stream_name: str,
        data_field: Union[bytes, str],
        data_dtype: Union[str, type, np.dtype],
        data_kwargs: dict,
        unit_ids: list,
        entry_ids: list[bytes],
        frames_per_entry: int = 1,
        timestamps: Optional[np.ndarray] = None,
        nsp_timestamps: Optional[np.ndarray] = None,
        t_start: Optional[float] = None,
        sampling_frequency: Optional[float] = None,
        unit_dim: int = 0,
        chunk_size: Optional[int] = None,
    ):
        """Initialize the RedisStreamRecordingSegment

        Parameters
        ----------
        client : redis.Redis
            Redis client connected to Redis server containing data
        stream_name : str
            Name of stream containing the recording data
        data_field : bytes or str
            Key or field within each Redis stream entry that
            contains the recording data
        data_dtype : str, type, or numpy.dtype
            The dtype of the data. Assumed to be a numeric type
            recognized by numpy, e.g. int8, float32, etc.
        unit_ids : list
            A list containing IDs for each unit
        entry_ids :  list of bytes
            A list containing the entry ID for each entry in the Redis
            stream in order
        frames_per_entry : int, default: 1
            Number of frames (i.e. a single time point) contained
            within each Redis stream entry
        timestamps : numpy.ndarray, optional
            A vector of timestamps corresponding to all samples in
            the segment
        t_start : float, optional
            The start time of the segment, relative to the recording
            start time
        sampling_frequency : float, optional
            The sampling frequency of the data in the segment
        channel_dim : int, default: 0
            If frames_per_entry > 1, then channel_dim indicates the
            axis ordering of the data in each entry. If channel_dim
            = 0, then the data is assumed to originally be of shape
            (channel_count, frames_per_entry). If channel_dim = 1,
            then the data is assumed to originally be of shape
            (frames_per_entry, channel_count)
        """
        # initialize base class
        BaseSortingSegment.__init__(self, t_start=t_start)

        # timestamp handling
        if sampling_frequency is None:
            assert timestamps is not None, "Pass either 'sampling_frequency' or 'timestamps'"
            assert timestamps.ndim == 1, "timestamps should be a 1D array"
        if timestamps is None:
            assert sampling_frequency is not None, "Pass either 'sampling_frequency' or 'timestamps'"
        self.time_vector = timestamps
        self.sampling_frequency = sampling_frequency

        # assign Redis client and check connection
        self._client = client
        self._client.ping()

        # arg checks
        assert unit_dim in [0, 1]
        assert len(entry_ids) == self._client.xlen(stream_name)

        # save some variables
        self._nsp_timestamps = nsp_timestamps
        self._stream_name = stream_name
        self._data_field = data_field
        self._data_dtype = data_dtype
        self._data_kwargs = data_kwargs
        self._unit_count = len(unit_ids)
        self._unit_ids = unit_ids
        self._entry_ids = entry_ids
        self._frames_per_entry = frames_per_entry
        self._num_samples = frames_per_entry * len(entry_ids)

        # make chunk size an init arg?
        self._spike_frames = None
        self._load_spike_frames(chunk_size=chunk_size)

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
                entry_data = read_entry(entry[1], self._data_field, dtype=self._data_dtype, **self._data_kwargs)
                assert entry_data.shape == (self._frames_per_entry, self._unit_count)
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
            stream_entries = self._client.xrange(self._stream_name, min=b"(" + stream_entries[-1][0], count=chunk_size)
        # stack all spike frames and labels
        spike_frames = np.concatenate(spike_frames, axis=0)
        spike_labels = np.concatenate(spike_labels, axis=0)
        # double check outputs
        assert len(spike_frames) == len(spike_labels)
        assert np.all((spike_labels >= 0) & (spike_labels < self._unit_count))
        # create list of views into spike frames indexed by label/unit
        self._spike_frames = [spike_frames[spike_labels == i] for i in range(self._unit_count)]

    def get_num_samples(self) -> int:
        return self._num_samples

    def get_entry_ids(self):
        return self._entry_ids

    def get_times(self):
        if self.time_vector is not None:
            if isinstance(self.time_vector, np.ndarray):
                return self.time_vector
            else:
                return np.array(self.time_vector)
        else:
            time_vector = np.arange(self.get_num_samples(), dtype="float64")
            time_vector /= self.sampling_frequency
            if self.t_start is not None:
                time_vector += self.t_start
            return time_vector

    def get_nsp_times(self):
        return self._nsp_timestamps

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

        return spike_frames