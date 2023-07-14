"""Class for converting generic ecephys data."""
import json
import redis
import numpy as np
from pynwb import NWBFile, TimeSeries
from typing import Optional, Union, Literal
from hdmf.backends.hdf5 import H5DataIO

from neuroconv.basetemporalalignmentinterface import BaseTemporalAlignmentInterface
from neuroconv.tools.nwb_helpers import get_module
from neuroconv.datainterfaces.ecephys.baserecordingextractorinterface import BaseRecordingExtractorInterface


class StaviskySpikingBandPowerInterface(BaseTemporalAlignmentInterface):
    """Spiking band power interface for Stavisky Redis conversion"""

    def __init__(
        self,
        port: int,
        host: str,
        stream_name: str,
        data_key: str,
        ts_key: str = "spiking_band_power",
        timestamp_source: str = "redis",
        timestamp_conversion: float = 1.0,
        timestamp_encoding: Optional[str] = None,
        timestamp_dtype: Optional[Union[str, type, np.dtype]] = None,
    ):
        """Initialize StaviskySpikingBandPowerInterface

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
        timestamp_source : bytes or str, default: "redis"
            The source of the timestamp information in the Redis stream.
            See `load_timestamps()`
        timestamp_conversion : float, default: 1
            Conversion factor needed to scale the timestamps to seconds.
            See `load_timestamps()`
        timestamp_encoding : {"string", "buffer"}, optional
            How timestamps are encoded in Redis entry data. See
            `load_timestamps()`
        timestamp_dtype : str, type, or numpy.dtype, optional
            The data type of the timestamps in Redis. See
            `load_timestamps()`
        """
        super().__init__(port=port, host=host, stream_name=stream_name, data_key=data_key)
        self.ts_key = ts_key
        self._timestamps = self.get_original_timestamps(
            timestamp_source=timestamp_source,
            timestamp_conversion=timestamp_conversion,
            timestamp_encoding=timestamp_encoding,
            timestamp_dtype=timestamp_dtype,
        )

    def get_original_timestamps(
        self,
        timestamp_source: str = "redis",
        timestamp_encoding: Optional[str] = None,
        timestamp_dtype: Optional[Union[str, type, np.dtype]] = None,
        timestamp_conversion: float = 1.0,
    ):
        """Get original timestamps for this data stream

        Parameters
        ----------
        timestamp_source : bytes or str, default: "redis"
            The source of the timestamp information in the Redis stream. If
            the timestamp source is "redis", the entry IDs are used as timestamps.
            Otherwise, the timestamp source is assumed to be a data key present
            in each entry
        timestamp_conversion : float, default: 1
            If `timestamp_source` is a Redis entry data key, then the user should
            provide the conversion factor needed to scale the timestamps to seconds
        timestamp_encoding : {"string", "buffer"}, optional
            If `timestamp_source` is a Redis entry data key, then how the
            timestamp is stored must be specified. If the encoding is "string",
            then the timestamp is treated as a human-readable string of some
            numeric type. If the encoding is "buffer", then the timestamp is
            treated as a raw byte buffer of some numeric type
        timestamp_dtype : str, type, or numpy.dtype, optional
            If `timestamp_source` is a Redis entry data key, then the data
            type of the timestamp must be specified. The provided data type
            is assumed to be a numeric type recognized by numpy, e.g. int8,
            float, float32, etc.
        """
        # Instantiate Redis client and check connection
        r = redis.Redis(
            port=self.source_data["port"],
            host=self.source_data["host"],
        )
        r.ping()
        stream_name = self.source_data["stream_name"]
        assert r.xlen(stream_name) > 0
        chunk_size = 1000  # TODO: figure out how to handle this

        # check timestamp-related args
        if timestamp_source == "redis":
            timestamp_conversion = 1e-3
        else:
            assert timestamp_encoding is not None
            assert timestamp_dtype is not None
            assert timestamp_conversion is not None
            timestamp_source = bytes(timestamp_source, "utf-8")
        if timestamp_encoding is not None:
            assert timestamp_encoding in ["str", "buffer"]
        assert timestamp_conversion > 0

        # extract timestamps
        timestamps = []
        # loop through stream entries
        stream_entries = r.xrange(stream_name, count=chunk_size)
        while len(stream_entries) > 0:
            for entry in stream_entries:
                # prepare timestamps
                if timestamp_source == "redis":
                    timestamps.append(float(str(entry[0], "utf-8").split("-")[0]))
                else:
                    if timestamp_encoding == "str":  # try direct casting for string encoded
                        ts = np.dtype(timestamp_dtype)(str(entry[1][timestamp_source], "utf-8"))
                        timestamps.append(ts.astype("float128", copy=False))
                    elif timestamp_encoding == "buffer":  # use np.frombuffer for byte buffer
                        ts = np.frombuffer(entry[1][timestamp_source], dtype=timestamp_dtype)
                        timestamps.append(ts.astype("float128", copy=False))
            # get next chunk of entries
            stream_entries = r.xrange(
                stream_name, min=b"(" + stream_entries[-1][0], count=chunk_size
            )  # '(' notation for exclusive indexing
        # make full arrays
        timestamps = np.array(timestamps, dtype=np.float128)

        # post-process timestamps
        timestamps *= timestamp_conversion

        # close redis client
        r.close()

        return timestamps

    def add_to_nwbfile(
        self,
        nwbfile: NWBFile,
        metadata: Optional[dict] = None,
        stub_test: bool = False,
        chunk_size: int = 1000,
        smooth_timestamps: bool = False,
    ):
        """Add spiking band power to NWB file

        Parameters
        ----------
        nwbfile : NWBFile, optional
            An NWBFile object to write to the location
        metadata : dict, optional
            Metadata dictionary with information used to create the NWBFile
        stub_test : bool, default: False
            Whether to only partially convert the data or not
        chunk_size : int, default: 1000
            The number of Redis entries to read from the stream
            per iteration
        smooth_timestamps : bool, default: False
            Whether to re-interpolate the timestamps to be
            regularly sampled or not
        """
        # Instantiate Redis client and check connection
        r = redis.Redis(
            port=self.source_data["port"],
            host=self.source_data["host"],
        )
        r.ping()
        stream_name = self.source_data["stream_name"]
        data_key = self.source_data["data_key"]
        data_key = data_key if isinstance(data_key, bytes) else bytes(data_key, "utf-8")
        assert r.xlen(stream_name) > 0

        # set max read len if stub_test
        max_len = r.xlen(stream_name) // 4 if stub_test else np.inf

        # get processing module
        module_name = "ecephys"
        module_description = "Contains processed ecephys data like spiking band power."
        processing_module = get_module(nwbfile=nwbfile, name=module_name, description=module_description)

        # extract sbp data
        sbp = []
        # loop through stream entries
        stream_entries = r.xrange(stream_name, count=chunk_size)
        while len(stream_entries) > 0 and len(sbp) < max_len:
            for entry in stream_entries:
                # read spiking band power data
                entry_sbp = np.frombuffer(entry[1][data_key], dtype=np.float32)
                sbp.append(entry_sbp)
            # get next chunk of entries
            stream_entries = r.xrange(
                stream_name, min=b"(" + stream_entries[-1][0], count=chunk_size
            )  # '(' notation for exclusive indexing
        # make full arrays
        sbp = np.stack(sbp, axis=0)

        timestamps = self.get_timestamps().astype("float64")
        if smooth_timestamps:
            timestamps = np.linspace(timestamps[0], timestamps[-1], len(timestamps))

        # get metadata about filtering, etc.
        bin_size = int(stream_name.split("_")[-1].strip("ms"))
        frequency = 1000 / bin_size
        data = json.loads(r.xrange("supergraph_stream")[0][1][b"data"])
        try:  # shouldn't fail but just in case
            params = data["nodes"]["featureExtraction"]["parameters"]
        except:
            params = {}
        butter_lowercut = params.get("butter_lowercut", None)
        butter_uppercut = params.get("butter_uppercut", None)
        butter_order = params.get("butter_order", None)
        clip_value = params.get("spike_pow_clip_thresh", None)

        # build description from metadata
        info = ""
        if butter_lowercut or butter_uppercut:
            info += (
                f" Bandpass filtered with a {butter_order or 'unknown'} order "
                f"Butterworth filter with lower cutoff {butter_lowercut or '0'} Hz "
                f"and upper cutoff {butter_uppercut or 'inf'} Hz."
            )
        if clip_value:
            info += f" Clipped to a maximum of {clip_value}."
        description = f"Spiking band power at {frequency} Hz." + info

        # create timeseries objs
        sbp_timeseries = TimeSeries(
            name=self.ts_key,
            data=H5DataIO(sbp, compression="gzip"),
            unit="V^2",
            conversion=1e-8,
            timestamps=H5DataIO(timestamps, compression="gzip"),
            description=description,
        )

        # add to nwbfile
        processing_module.add_data_interface(sbp_timeseries)

        # close redis client
        r.close()

        return nwbfile

    def get_timestamps(self) -> np.ndarray:
        return self._timestamps

    def set_aligned_timestamps(self, aligned_timestamps: np.ndarray) -> None:
        self._timestamps = aligned_timestamps
