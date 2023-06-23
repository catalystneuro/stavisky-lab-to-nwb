"""Class for converting generic ecephys data."""
import redis
import numpy as np
from pynwb import NWBFile, TimeSeries
from typing import Optional, Union, Literal

from neuroconv.basedatainterface import BaseDataInterface
from neuroconv.tools.nwb_helpers import get_module
from neuroconv.datainterfaces.ecephys.baserecordingextractorinterface import BaseRecordingExtractorInterface


class StaviskySpikingBandPowerInterface(BaseDataInterface):
    """Spiking band power interface for Stavisky Redis conversion"""

    # TODO: if we decide timestamps need to be aligned with sorting, etc. then
    # may need to subclass from BaseTemporalAlignmentInterface instead

    def __init__(
        self,
        port: int,
        host: str,
    ):
        super().__init__(port=port, host=host)

    def add_to_nwbfile(
        self,
        nwbfile: NWBFile,
        metadata: Optional[dict] = None,
        stub_test: bool = False,
        chunk_size: int = 1000,
        timestamp_source: str = "redis",
        timestamp_encoding: Optional[str] = None,
        timestamp_dtype: Optional[Union[str, type, np.dtype]] = None,
        timestamp_unit: Optional[str] = None,
        smooth_timestamps: bool = False,
        start_time: Optional[float] = None,
    ):
        # Instantiate Redis client and check connection
        r = redis.Redis(
            port=self.source_data["port"],
            host=self.source_data["host"],
        )
        r.ping()

        # check timestamp-related args
        if timestamp_source == "redis":
            timestamp_unit = "ms"
            start_time = np.frombuffer(r.xrange("metadata")[0][1][b"startTime"], dtype=np.float64).item()
        else:
            assert timestamp_encoding is not None
            assert timestamp_dtype is not None
            assert timestamp_unit is not None
            timestamp_source = bytes(timestamp_source, "utf-8")
        if timestamp_encoding is not None:
            assert timestamp_encoding in ["str", "buffer"]
        if timestamp_unit is not None:
            assert timestamp_unit in ["s", "ms", "us"]  # TODO: or just directly use scaling factor

        # set max read len if stub_test
        max_len_1ms = 100000 if stub_test else np.inf  # about 1/4 of the data
        max_len_20ms = 5000 if stub_test else np.inf

        # get processing module
        module_name = "ecephys"
        module_description = "Contains processed ecephys data like spiking band power"
        processing_module = get_module(nwbfile=nwbfile, name=module_name, description=module_description)

        # extract 1 ms data
        sbp_1ms = []
        timestamps_1ms = []
        # loop through stream entries
        stream_entries = r.xrange("neuralFeatures_1ms", count=chunk_size)
        while len(stream_entries) > 0 and len(sbp_1ms) < max_len_1ms:
            for entry in stream_entries:
                # read spiking band power data
                entry_sbp = np.frombuffer(entry[1][b"spike_band_power"], dtype=np.float32)
                sbp_1ms.append(entry_sbp)
                # prepare timestamps
                if timestamp_source == "redis":
                    timestamps_1ms.append(float(str(entry[0], "utf-8").split("-")[0]))
                else:
                    if timestamp_encoding == "str":  # try direct casting for string encoded
                        ts = np.dtype(timestamp_dtype)(str(entry[1][timestamp_source], "utf-8"))
                        timestamps_1ms.append(ts.astype("float64", copy=False))
                    elif timestamp_encoding == "buffer":  # use np.frombuffer for byte buffer
                        ts = np.frombuffer(entry[1][timestamp_source], dtype=timestamp_dtype)
                        timestamps_1ms.append(ts.astype("float64", copy=False))
            # get next chunk of entries
            stream_entries = r.xrange(
                "neuralFeatures_1ms", min=b"(" + stream_entries[-1][0], count=chunk_size
            )  # '(' notation for exclusive indexing
        # make full arrays
        sbp_1ms = np.stack(sbp_1ms, axis=0)
        timestamps_1ms = np.array(timestamps_1ms)

        # extract 20 ms re-binned data also (at their request)
        sbp_20ms = []
        timestamps_20ms = []
        # loop through stream entries
        stream_entries = r.xrange("binnedFeatures_20ms", count=chunk_size)
        while len(stream_entries) > 0 and len(sbp_20ms) < max_len_20ms:
            for entry in stream_entries:
                # read spiking band power data
                entry_sbp = np.frombuffer(entry[1][b"spike_band_power_bin"], dtype=np.float32)
                sbp_20ms.append(entry_sbp)
                # prepare timestamps
                if timestamp_source == "redis":
                    timestamps_20ms.append(float(str(entry[0], "utf-8").split("-")[0]))
                else:
                    if timestamp_encoding == "str":  # try direct casting for string encoded
                        ts = np.dtype(timestamp_dtype)(str(entry[1][timestamp_source], "utf-8"))
                        timestamps_20ms.append(ts.astype("float64", copy=False))
                    elif timestamp_encoding == "buffer":  # use np.frombuffer for byte buffer
                        ts = np.frombuffer(entry[1][timestamp_source], dtype=timestamp_dtype)
                        timestamps_20ms.append(ts.astype("float64", copy=False))
            # get next chunk of entries
            stream_entries = r.xrange(
                "binnedFeatures_20ms", min=b"(" + stream_entries[-1][0], count=chunk_size
            )  # '(' notation for exclusive indexing
        # make full arrays
        sbp_20ms = np.stack(sbp_20ms, axis=0)
        timestamps_20ms = np.array(timestamps_20ms)

        # post-process timestamps
        if timestamp_unit == "ms":
            timestamps_1ms *= 1e-3
            timestamps_20ms *= 1e-3
        elif timestamp_unit == "us":
            timestamps_1ms *= 1e-6
            timestamps_20ms *= 1e-6
        if smooth_timestamps:
            timestamps_1ms = np.linspace(timestamps_1ms[0], timestamps_1ms[-1], len(timestamps_1ms))
            timestamps_20ms = np.linspace(timestamps_20ms[0], timestamps_20ms[-1], len(timestamps_20ms))

        # subtract start time
        if start_time is None:
            start_time = timestamps_1ms[0]
        timestamps_1ms -= start_time
        timestamps_20ms -= start_time

        # create timeseries objs
        sbp_1ms_timeseries = TimeSeries(
            name="spiking_band_power_1ms",
            data=sbp_1ms,
            unit="V^2/Hz",
            # conversion=?, # TODO: double check scaling of values
            timestamps=timestamps_1ms,
            description="Spiking band power in the 250 Hz to 5 kHz frequency range computed 1 kHz",
        )
        sbp_20ms_timeseries = TimeSeries(
            name="spiking_band_power_20ms",
            data=sbp_20ms,
            unit="V^2/Hz",
            # conversion=?,
            timestamps=timestamps_20ms,
            description="Spiking band power in the 250 Hz to 5 kHz frequency range computed 1 kHz and re-binned to 50 Hz",
        )

        # add to nwbfile
        processing_module.add_data_interface(sbp_1ms_timeseries)
        processing_module.add_data_interface(sbp_20ms_timeseries)

        # close redis client
        r.close()

        # import pdb; pdb.set_trace()

        return nwbfile
