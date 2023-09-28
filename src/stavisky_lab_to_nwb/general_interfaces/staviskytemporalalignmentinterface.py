import json
import redis
import numpy as np
from pynwb import TimeSeries
from pynwb.core import MultiContainerInterface
from typing import Optional, Union, Literal
from hdmf.backends.hdf5 import H5DataIO
from abc import abstractmethod

from neuroconv.basetemporalalignmentinterface import BaseTemporalAlignmentInterface

from stavisky_lab_to_nwb.utils.redis import read_stream_fields, RedisDataChunkIterator
from stavisky_lab_to_nwb.utils.timestamps import get_stream_ids_and_timestamps, smooth_timestamps


class DualTimestampTemporalAlignmentInterface(BaseTemporalAlignmentInterface):
    """Abstract base class defining interface for dual-timestamp alignment functionality"""

    @abstractmethod
    def get_original_timestamps(self) -> tuple[list[bytes], np.ndarray, Optional[np.ndarray]]:
        raise NotImplementedError(
            "Unable to retrieve the original unaltered timestamps for this interface! "
            "Define the `get_original_timestamps` method for this interface."
        )

    @abstractmethod
    def get_timestamps(self, nsp: bool = False) -> np.ndarray:
        raise NotImplementedError(
            "Unable to retrieve timestamps for this interface! Define the `get_timestamps` method for this interface."
        )

    @abstractmethod
    def set_aligned_timestamps(self, aligned_timestamps: np.ndarray, nsp: bool = False) -> None:
        raise NotImplementedError(
            "The protocol for synchronizing the timestamps of this interface has not been specified!"
        )

    @abstractmethod
    def get_entry_ids(self):
        raise NotImplementedError("The protocol for getting entry IDs of this interface has not been specified!")

    def set_aligned_starting_time(self, aligned_starting_time: float, nsp: bool = False) -> None:
        self.set_aligned_timestamps(aligned_timestamps=self.get_timestamps(nsp=nsp) + aligned_starting_time, nsp=nsp)

    def align_by_interpolation(
        self, unaligned_timestamps: np.ndarray, aligned_timestamps: np.ndarray, nsp: bool = False
    ) -> None:
        self.set_aligned_timestamps(
            aligned_timestamps=np.interp(
                x=self.get_timestamps(nsp=nsp).astype("float64"),
                xp=unaligned_timestamps.astype("float64"),
                fp=aligned_timestamps.astype("float64"),
            ),
            nsp=nsp,
        )

    def set_dtype(self, dtype, nsp: bool = False):
        self.set_aligned_timestamps(self.get_timestamps(nsp=nsp).astype(dtype))


class StaviskyTemporalAlignmentInterface(DualTimestampTemporalAlignmentInterface):
    """Base class for interfaces to write data to NWB with two different time bases"""

    default_data_kwargs: dict = dict()

    def __init__(
        self,
        port: int,
        host: str,
        stream_name: str,
        data_field: str,
        ts_key: str,
        frames_per_entry: int = 1,
        data_dtype: Optional[str] = None,
        data_kwargs: dict = dict(),
        nsp_timestamp_field: Optional[str] = None,
        nsp_timestamp_conversion: Optional[float] = None,
        nsp_timestamp_encoding: Optional[str] = None,
        nsp_timestamp_dtype: Optional[Union[str, type, np.dtype]] = None,
        nsp_timestamp_index: Optional[int] = None,
        smoothing_kwargs: dict = {},
        load_timestamps: bool = True,
        chunk_size: Optional[int] = None,
    ):
        """Base class for dual timestamp handling

        Parameters
        ----------
        port : int
            Port number for Redis server
        host : str
            Host name for Redis server, e.g. "localhost"
        stream_name : str
            Name of stream containing the spiking band power data
        data_key : str
            Key or field within each Redis stream entry that
            contains the spiking band power data
        ts_key : str
            Name of the timeseries to be saved to NWB file
        data_kwargs : dict
            Read kwargs for the data
        frames_per_entry : int, default: 1
            Number of separate timesteps included in each entry,
            default 1
        nsp_timestamp_field : str, optional
            The source of the nsp timestamp information in the Redis stream.
            See `get_original_timestamps()`
        nsp_timestamp_conversion : float, default: 1
            Conversion factor needed to scale the timestamps to seconds.
            See `get_original_timestamps()`
        nsp_timestamp_encoding : {"string", "buffer"}, optional
            How timestamps are encoded in Redis entry data. See
            `get_original_timestamps()`
        nsp_timestamp_dtype : str, type, or numpy.dtype, optional
            The data type of the timestamps in Redis. See
            `get_original_timestamps()`
        nsp_timestamp_index : int, optional
            Index of timestamp in timestamp arrays. See `get_original_timestamps()`
        """
        super().__init__(port=port, host=host, stream_name=stream_name, data_field=data_field)
        self.ts_key = ts_key
        self.data_kwargs = self.default_data_kwargs.copy()
        if "shape" not in self.data_kwargs:
            self.data_kwargs["shape"] = (frames_per_entry, -1)  # doesn't handle transposed data!
        if data_dtype is not None:
            data_kwargs["dtype"] = data_dtype
        self.data_kwargs.update(data_kwargs)
        self.chunk_size = chunk_size
        if load_timestamps:
            self._entry_ids, self._timestamps, self._nsp_timestamps = self.get_original_timestamps(
                frames_per_entry=frames_per_entry,
                nsp_timestamp_field=nsp_timestamp_field,
                nsp_timestamp_conversion=nsp_timestamp_conversion,
                nsp_timestamp_encoding=nsp_timestamp_encoding,
                nsp_timestamp_dtype=nsp_timestamp_dtype,
                nsp_timestamp_index=nsp_timestamp_index,
                smoothing_kwargs=smoothing_kwargs,
            )
        else:
            self._entry_ids = None
            self._timestamps = None
            self._nsp_timestamps = None

    def get_original_timestamps(
        self,
        frames_per_entry: int = 1,
        nsp_timestamp_field: Optional[str] = None,
        nsp_timestamp_encoding: Optional[str] = None,
        nsp_timestamp_dtype: Optional[Union[str, type, np.dtype]] = None,
        nsp_timestamp_conversion: Optional[float] = None,
        nsp_timestamp_index: Optional[int] = None,
        smoothing_kwargs: dict = {},
    ):
        """Get original timestamps for this data stream

        Parameters
        ----------
        nsp_timestamp_source : bytes or str, optional
            The source of the timestamp information in the Redis stream. By
            default, Redis timestamps are always read. If specified,
            the timestamp source is assumed to be a data key present
            in each entry and is saved as `_alt_timestamps`
        nsp_timestamp_conversion : float, default: 1
            If `nsp_timestamp_source` is a Redis entry data key, then the user should
            provide the conversion factor needed to scale the timestamps to seconds
        nsp_timestamp_encoding : {"string", "buffer"}, optional
            If `nsp_timestamp_source` is a Redis entry data key, then how the
            timestamp is stored must be specified. If the encoding is "string",
            then the timestamp is treated as a human-readable string of some
            numeric type. If the encoding is "buffer", then the timestamp is
            treated as a raw byte buffer of some numeric type
        nsp_timestamp_dtype : str, type, or numpy.dtype, optional
            If `nsp_timestamp_source` is a Redis entry data key, then the data
            type of the timestamp must be specified. The provided data type
            is assumed to be a numeric type recognized by numpy, e.g. int8,
            float, float32, etc.
        nsp_timestamp_index : int, optional
            If the timestamp field to read from has multiple values
            for each entry, then this field is used to index
            into that array
        """
        # Instantiate Redis client and check connection
        r = redis.Redis(
            port=self.source_data["port"],
            host=self.source_data["host"],
        )
        r.ping()

        # read timestamp data from redis
        entry_ids, redis_timestamps, nsp_timestamps = get_stream_ids_and_timestamps(
            client=r,
            stream_name=self.source_data["stream_name"],
            frames_per_entry=frames_per_entry,
            timestamp_field=nsp_timestamp_field,
            timestamp_conversion=nsp_timestamp_conversion,
            timestamp_encoding=nsp_timestamp_encoding,
            timestamp_dtype=nsp_timestamp_dtype,
            timestamp_index=nsp_timestamp_index,
            chunk_size=self.chunk_size,
        )
        if smoothing_kwargs:
            redis_timestamps = smooth_timestamps(
                redis_timestamps,
                frames_per_entry=frames_per_entry,
                **smoothing_kwargs,
            )

        # close redis client
        r.close()

        return entry_ids, redis_timestamps, nsp_timestamps

    def get_timestamps(self, nsp: bool = False) -> np.ndarray:
        return self._nsp_timestamps if nsp else self._timestamps

    def set_aligned_timestamps(self, aligned_timestamps: np.ndarray, nsp: bool = False) -> None:
        if aligned_timestamps is None:
            raise ValueError("Cannot set aligned timestamps with `aligned_timestamps=None`")
        if nsp:
            if self._nsp_timestamps is None:
                raise AssertionError("Original NSP timestamps were not set before calling `set_aligned_timestamps()`")
            if len(aligned_timestamps) != len(self._nsp_timestamps):
                print("Warning: length of aligned timestamps != length of original timestamps")
            self._nsp_timestamps = aligned_timestamps
        else:
            if len(aligned_timestamps) != len(self._timestamps):
                print("Warning: length of aligned timestamps != length of original timestamps")
            self._timestamps = aligned_timestamps

    def get_entry_ids(self):
        return self._entry_ids

    def set_timestamps_from_interface(
        self,
        interface: BaseTemporalAlignmentInterface,
    ):
        self._timestamps = interface.get_timestamps()
        if isinstance(interface, DualTimestampTemporalAlignmentInterface):
            self._nsp_timestamps = interface.get_timestamps(nsp=True)
            self._entry_ids = interface.get_entry_ids()

    def get_data_iterator(
        self,
        client: redis.Redis,
        stub_test: bool = False,
        use_chunk_iterator: bool = False,
        iterator_opts: dict = {},
        chunk_size: Optional[int] = None,
    ):
        # Instantiate Redis client and check connection
        stream_name = self.source_data["stream_name"]
        data_field = self.source_data["data_field"]
        assert client.xlen(stream_name) > 0

        # set max read len if stub_test
        max_len = client.xlen(stream_name) // 4 if stub_test else np.inf

        # make iterators
        if not use_chunk_iterator:
            data_dict = read_stream_fields(
                client=client,
                stream_name=stream_name,
                field_kwargs={data_field: self.data_kwargs},
                return_ids=False,
                chunk_size=(chunk_size or self.chunk_size),
                max_stream_len=max_len,
            )
            iterator = np.concatenate(data_dict[data_field], axis=0)
        else:
            assert self._entry_ids is not None, f"Must load entry ids to use `RedisDataChunkIterator`"
            iterator = RedisDataChunkIterator(
                client=client,
                stream_name=stream_name,
                field=data_field,
                entry_ids=self._entry_ids,
                read_kwargs=self.data_kwargs,
                max_stream_len=max_len,
                **iterator_opts,
            )
        return iterator

    def add_to_processing_module(
        self,
        processing_module,
        data: Union[np.ndarray, RedisDataChunkIterator],
        dataclass=TimeSeries,
        dataclass_kwargs: dict = {},
        containerclass: Optional[MultiContainerInterface] = None,
        container_name: Optional[str] = None,
        stub_test: bool = False,
        save_dual_timestamps: bool = True,
    ):
        # check kwargs
        if "name" in dataclass_kwargs:
            print("Ignoring `name` in `dataclass_kwargs`, using `ts_key` instead.")
            dataclass_kwargs.pop("name")

        # get and truncate timestamps
        timestamps = self.get_timestamps()
        assert timestamps is not None, "Timestamps must be loaded before calling `add_to_processing_module()`"
        nsp_timestamps = self.get_timestamps(nsp=True)
        data_len = data._get_maxshape()[0] if isinstance(data, RedisDataChunkIterator) else data.shape[0]
        if stub_test:
            timestamps = timestamps[:data_len]
            if nsp_timestamps is not None:
                nsp_timestamps = nsp_timestamps[:data_len]
        assert len(timestamps) == data_len, "Timestamps and data have different lengths!"
        if nsp_timestamps is not None:
            assert len(nsp_timestamps) == data_len, "Timestamps and data have different lengths!"

        # create timeseries objs
        data_to_add = []
        if nsp_timestamps is None or not save_dual_timestamps:
            nwb_data = dataclass(
                name=self.ts_key,
                data=H5DataIO(data, compression="gzip"),
                timestamps=H5DataIO(timestamps, compression="gzip"),
                **dataclass_kwargs,
            )
            data_to_add.append(nwb_data)
        else:
            redis_dataclass_kwargs = dataclass_kwargs.copy()
            redis_dataclass_kwargs["description"] = (
                redis_dataclass_kwargs["description"].strip() + " Aligned with real-time packet write time."
            )
            redis_nwb_data = dataclass(
                name=self.ts_key + "_realtime",
                data=H5DataIO(data, compression="gzip"),
                timestamps=H5DataIO(timestamps, compression="gzip"),
                **redis_dataclass_kwargs,
            )
            data_to_add.append(redis_nwb_data)
            nsp_dataclass_kwargs = dataclass_kwargs.copy()
            nsp_dataclass_kwargs["description"] = (
                nsp_dataclass_kwargs["description"].strip()
                + " Aligned with recording nsp timestamps at the start of each bin."
            )
            nsp_nwb_data = dataclass(
                name=self.ts_key + "_nsp",
                data=redis_nwb_data,
                timestamps=H5DataIO(nsp_timestamps, compression="gzip"),
                **nsp_dataclass_kwargs,
            )
            data_to_add.append(nsp_nwb_data)

        # add to container if provided
        if containerclass is not None:
            if container_name in processing_module.data_interfaces:
                container = processing_module.data_interfaces[container_name]
            else:
                container = containerclass(name=container_name)
                processing_module.add(container)
            container_add = getattr(container, container.__clsconf__.get("add"))
            for data in data_to_add:
                container_add(data)
        # else add to proc module
        else:
            processing_module.add(data_to_add)
