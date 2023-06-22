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
    
    def __init__(
        self,
        port: int,
        host: str,
        timestamp_source: str = "redis",
        timestamp_encoding: Optional[str] = None,
        timestamp_dtype: Optional[Union[str, type, np.dtype]] = None,
        timestamp_unit: Optional[str] = None, # Literal["s", "ms", "us"]
        start_time: Optional[float] = None,
    ):
        super().__init__(port=port, host=host)
        if timestamp_source == "redis":
            timestamp_unit = "ms"
        else:
            assert (timestamp_encoding is not None)
            assert (timestamp_dtype is not None)
            assert (timestamp_unit is not None)
            timestamp_source = bytes(timestamp_source, "utf-8")
        if timestamp_encoding is not None:
            assert timestamp_encoding in ["str", "buffer"]
        self._timestamp_source = timestamp_source
        self._timestamp_encoding = timestamp_encoding
        self._timestamp_dtype = timestamp_dtype
        self._timestamp_unit = timestamp_unit
        self._start_time = start_time
        self._recording = None
        self._recording_frequency_ratio_1ms = None
            
    def get_metadata(self):
        # Automatically retrieve as much metadata as possible
        metadata = super().get_metadata()
        
        return metadata
    
    def register_recording(
        self,
        recording: BaseRecordingExtractorInterface,
        frequency_ratio: int = 1,
    ):
        self._recording = recording
        self._recording_frequency_ratio_1ms = frequency_ratio

    def add_to_nwbfile(
        self,
        nwbfile: NWBFile,
        metadata: Optional[dict] = None,
        stub_test: bool = False,
        chunk_size: int = 1000,
    ):
        # Instantiate Redis client and check connection
        r = redis.Redis(
            port=self.source_data["port"],
            host=self.source_data["host"],
        )
        r.ping()

        # get processing module
        module_name = "ecephys"
        module_description = "Contains processed ecephys data like spiking band power"
        processing_module = get_module(nwbfile=nwbfile, name=module_name, description=module_description)
        
        # prepare timestamps if provided
        build_timestamps = self._recording is None
        if build_timestamps:
            timestamps_1ms = []
            timestamps_20ms = []
        else:
            base_timestamps = self._recording.recording_extractor.get_times() # assumes 1 segment
            timestamps_1ms = base_timestamps[::self._recording_frequency_ratio_1ms]
            timestamps_20ms = timestamps_1ms[:-19:20]
        
        # extract 1 ms data
        sbp_1ms = []
        
        stream_entries = r.xrange('neuralFeatures_1ms', count=chunk_size)
        while len(stream_entries) > 0:
            for entry in stream_entries:
                entry_sbp = np.frombuffer(entry[1][b'spike_band_power'], dtype=np.float32)
                sbp_1ms.append(entry_sbp)
                if build_timestamps:
                    if self._timestamp_source == "redis":
                        timestamps_1ms.append(float(str(entry[0], "UTF-8").split("-")[0]))
                    else:
                        if self._timestamp_encoding == "str":
                            ts = np.dtype(self._timestamp_dtype)(str(entry[1][self._timestamp_source], "UTF-8"))
                            timestamps_1ms.append(ts.astype('float64', copy=False))
                        elif self._timestamp_encoding == "buffer":
                            ts = np.frombuffer(entry[1][self._timestamp_source], dtype=self._timestamp_dtype)
                            timestamps_1ms.append(ts.astype('float64', copy=False))
            last_id = stream_entries[-1][0]
            sub_ms_identifier = int(last_id.split(b'-')[-1])
            next_id = last_id.replace(
                b'-' + bytes(str(sub_ms_identifier), 'UTF-8'),
                b'-' + bytes(str(sub_ms_identifier + 1), 'UTF-8')
            )
            stream_entries = r.xrange('neuralFeatures_1ms', min=next_id, count=chunk_size)
        sbp_1ms = np.stack(sbp_1ms, axis=0)
        if build_timestamps:
            timestamps_1ms = np.array(timestamps_1ms)
        
        # extract 20 ms re-binned data also (at their request)
        sbp_20ms = []
        
        stream_entries = r.xrange('binnedFeatures_20ms', count=chunk_size)
        while len(stream_entries) > 0:
            for entry in stream_entries:
                entry_sbp = np.frombuffer(entry[1][b'spike_band_power_bin'], dtype=np.float32)
                # TODO: use BRAND_time timestamps?
                sbp_20ms.append(entry_sbp)
                if build_timestamps:
                    if self._timestamp_source == "redis":
                        timestamps_20ms.append(float(str(entry[0], "UTF-8").split("-")[0]))
                    else:
                        if self._timestamp_encoding == "str":
                            ts = np.dtype(self._timestamp_dtype)(str(entry[1][self._timestamp_source], "UTF-8"))
                            timestamps_20ms.append(ts.astype('float64', copy=False))
                        elif self._timestamp_encoding == "buffer":
                            ts = np.frombuffer(entry[1][self._timestamp_source], dtype=self._timestamp_dtype)
                            timestamps_20ms.append(ts.astype('float64', copy=False))
            last_id = stream_entries[-1][0]
            sub_ms_identifier = int(last_id.split(b'-')[-1])
            next_id = last_id.replace(
                b'-' + bytes(str(sub_ms_identifier), 'UTF-8'),
                b'-' + bytes(str(sub_ms_identifier + 1), 'UTF-8')
            )
            stream_entries = r.xrange('binnedFeatures_20ms', min=next_id, count=chunk_size)
        sbp_20ms = np.stack(sbp_20ms, axis=0)
        if build_timestamps:
            timestamps_20ms = np.array(timestamps_20ms)
        
        # TODO: use timestamp smoothing?
        if build_timestamps:
            if self._timestamp_unit == "ms":
                timestamps_1ms *= 1e-3
                timestamps_20ms *= 1e-3
            elif self._timestamp_unit == "us":
                timestamps_1ms *= 1e-6
                timestamps_20ms *= 1e-6
        
        # subtract start time if not using recording timestamps
        if build_timestamps:
            if self._timestamp_source == "redis":
                start_time = np.frombuffer(r.xrange("metadata")[0][1][b"startTime"], dtype=np.float64).item()
            elif self._start_time is None:
                start_time = timestamps_1ms[0]
            timestamps_1ms -= start_time
            timestamps_20ms -= start_time
        
        # create timeseries objs
        sbp_1ms_timeseries = TimeSeries(
            name="spiking_band_power_1ms",
            data=sbp_1ms,
            unit="V^2/Hz",
            timestamps=timestamps_1ms,
            description="Spiking band power in the 250 Hz to 5 kHz frequency range computed 1 kHz",
        )
        sbp_20ms_timeseries = TimeSeries(
            name="spiking_band_power_20ms",
            data=sbp_20ms,
            unit="V^2/Hz",
            timestamps=timestamps_20ms,
            description="Spiking band power in the 250 Hz to 5 kHz frequency range computed 1 kHz and re-binned to 50 Hz",
        )
        
        # add to nwbfile
        processing_module.add_data_interface(sbp_1ms_timeseries)
        processing_module.add_data_interface(sbp_20ms_timeseries)
        
        # close redis client
        r.close()
        
        return nwbfile

        