"""Class for converting generic ecephys data."""
import json
import redis
import numpy as np
from pynwb import NWBFile, TimeSeries
from pynwb.ecephys import FilteredEphys, ElectricalSeries
from typing import Optional, Union, Literal
from hdmf.backends.hdf5 import H5DataIO

from neuroconv.basetemporalalignmentinterface import BaseTemporalAlignmentInterface
from neuroconv.tools.nwb_helpers import get_module
from neuroconv.datainterfaces.ecephys.baserecordingextractorinterface import BaseRecordingExtractorInterface

from stavisky_lab_to_nwb.general_interfaces.staviskytemporalalignmentinterface import StaviskyTemporalAlignmentInterface


class StaviskySpikingBandPowerInterface(StaviskyTemporalAlignmentInterface):
    """Spiking band power interface for Stavisky Redis conversion"""
    default_data_kwargs: dict = dict(dtype="float32", encoding="buffer", shape=(1,256))

    def __init__(
        self,
        port: int,
        host: str,
        stream_name: str,
        data_field: str,
        ts_key: str = "spiking_band_power",
        frames_per_entry: int = 1,
        data_dtype: Optional[str] = None,
        data_kwargs: dict = dict(),
        nsp_timestamp_field: Optional[str] = "input_nsp_timestamp",
        nsp_timestamp_conversion: Optional[float] = 1.0 / 3.0e4,
        nsp_timestamp_encoding: Optional[str] = "buffer",
        nsp_timestamp_dtype: Optional[Union[str, type, np.dtype]] = "int64",
        nsp_timestamp_index: Optional[int] = 0,
        load_timestamps: bool = True,
        chunk_size: Optional[int] = None,
    ):
        super().__init__(
            port=port, 
            host=host, 
            stream_name=stream_name, 
            data_field=data_field,
            ts_key=ts_key,
            frames_per_entry=frames_per_entry,
            data_dtype=data_dtype,
            data_kwargs=data_kwargs,
            nsp_timestamp_field=nsp_timestamp_field,
            nsp_timestamp_conversion=nsp_timestamp_conversion,
            nsp_timestamp_encoding=nsp_timestamp_encoding,
            nsp_timestamp_dtype=nsp_timestamp_dtype,
            nsp_timestamp_index=nsp_timestamp_index,
            load_timestamps=load_timestamps,
            chunk_size=chunk_size,
        )

    def add_to_nwbfile(
        self,
        nwbfile: NWBFile,
        metadata: Optional[dict] = None,
        stub_test: bool = False,
        chunk_size: Optional[int] = None,
        use_chunk_iterator: bool = False,
        iterator_opts: dict = dict(),
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
        chunk_size : int, optional
            The number of Redis entries to read from the stream
            per iteration
        """
        # Instantiate Redis client and check connection
        r = redis.Redis(
            port=self.source_data["port"],
            host=self.source_data["host"],
        )
        r.ping()
        
        # read data
        sbp = self.get_data_iterator(
            client=r, 
            stub_test=stub_test, 
            chunk_size=chunk_size,
            use_chunk_iterator=use_chunk_iterator,
            iterator_opts=iterator_opts,
        )

        # get metadata about filtering, etc.
        bin_size = int(self.source_data["stream_name"].split("_")[-1].strip("ms"))
        frequency = 1000 / bin_size
        data = json.loads(r.xrange("supergraph_stream")[0][1][b"data"])
        try:  # shouldn't fail but just in case
            params = data["nodes"]["featureExtraction_and_binning"]["parameters"]
        except Exception as e:
            print(f"Unable to extract filtering info: {e}")
            params = {}
        butter_lowercut = params.get("butter_lowercut", None)
        butter_uppercut = params.get("butter_uppercut", None)
        butter_order = params.get("butter_order", None)
        clip_value = params.get("spike_pow_clip_thresh", None)

        # build description from metadata
        info = ""
        if butter_lowercut or butter_uppercut:
            info += (
                f"Bandpass filtered with a {butter_order or 'unknown'} order "
                f"Butterworth filter with lower cutoff {butter_lowercut or '0'} Hz "
                f"and upper cutoff {butter_uppercut or 'inf'} Hz."
            )
        if clip_value:
            info += (" " if info else "") + f"Clipped to a maximum of {clip_value}."
        description = f"Spiking band power at {frequency} Hz." + (" " if info else "") + info

        # get processing module
        module_name = "ecephys"
        module_description = "Intermediate data from extracellular electrophysiology recordings, e.g., LFP." 
        processing_module = get_module(nwbfile=nwbfile, name=module_name, description=module_description)
        
        # add to nwbfile
        dataclass_kwargs = dict(
            unit="V^2",
            conversion=1e-8,
            description=description,
        )
        self.add_to_processing_module(
            processing_module=processing_module,
            data=sbp,
            dataclass=TimeSeries,
            dataclass_kwargs=dataclass_kwargs,
            stub_test=stub_test,
        )
        
        # close redis client
        r.close()

        return nwbfile


class StaviskyFilteredRecordingInterface(StaviskyTemporalAlignmentInterface):
    """Filtered continuous data interface for Stavisky conversion"""
    default_data_kwargs: dict = dict(dtype="int16", encoding="buffer", shape=(300,256))
    
    def __init__(
        self,
        port: int,
        host: str,
        stream_name: str,
        data_field: str,
        ts_key: str = "filtered_ephys",
        frames_per_entry: int = 300,
        data_dtype: Optional[str] = None,
        data_kwargs: dict = dict(),
        nsp_timestamp_field: Optional[str] = "timestamps",
        nsp_timestamp_conversion: Optional[float] = 1.0 / 3.0e4,
        nsp_timestamp_encoding: Optional[str] = "buffer",
        nsp_timestamp_dtype: Optional[Union[str, type, np.dtype]] = "int64",
        nsp_timestamp_index: Optional[int] = None,
        load_timestamps: bool = True,
        chunk_size: Optional[int] = None,
    ):
        super().__init__(
            port=port, 
            host=host, 
            stream_name=stream_name, 
            data_field=data_field,
            ts_key=ts_key,
            frames_per_entry=frames_per_entry,
            data_dtype=data_dtype,
            data_kwargs=data_kwargs,
            nsp_timestamp_field=nsp_timestamp_field,
            nsp_timestamp_conversion=nsp_timestamp_conversion,
            nsp_timestamp_encoding=nsp_timestamp_encoding,
            nsp_timestamp_dtype=nsp_timestamp_dtype,
            nsp_timestamp_index=nsp_timestamp_index,
            load_timestamps=load_timestamps,
            chunk_size=chunk_size,
        )
        
    def add_to_nwbfile(
        self,
        nwbfile: NWBFile,
        metadata: Optional[dict] = None,
        stub_test: bool = False,
        chunk_size: Optional[int] = None,
        use_chunk_iterator: bool = False,
        iterator_opts: dict = dict(),
    ):
        # Instantiate Redis client and check connection
        r = redis.Redis(
            port=self.source_data["port"],
            host=self.source_data["host"],
        )
        r.ping()
        
        # read data
        filt_ephys = self.get_data_iterator(
            client=r, 
            stub_test=stub_test, 
            chunk_size=chunk_size,
            use_chunk_iterator=use_chunk_iterator,
            iterator_opts=iterator_opts,
        )
        
        # get filtering info
        data = json.loads(r.xrange("supergraph_stream")[0][1][b"data"])
        try:  # shouldn't fail but just in case
            params = data["nodes"]["featureExtraction_and_binning"]["parameters"]
        except Exception as e:
            print(f"Unable to extract filtering info: {e}")
            params = {}
        butter_lowercut = params.get("butter_lowercut", None)
        butter_uppercut = params.get("butter_uppercut", None)
        butter_order = params.get("butter_order", None)

        # build description from metadata
        info = ""
        if butter_lowercut or butter_uppercut:
            info += (
                f"Bandpass filtered with a {butter_order or 'unknown'} order "
                f"Butterworth filter with lower cutoff {butter_lowercut or '0'} Hz "
                f"and upper cutoff {butter_uppercut or 'inf'} Hz."
            )
        description = f"Filtered continuous ecephys data."
        
        # get processing module
        module_name = "ecephys"
        module_description = "Intermediate data from extracellular electrophysiology recordings, e.g., LFP." 
        processing_module = get_module(nwbfile=nwbfile, name=module_name, description=module_description)
        
        # add to nwbfile
        table_ids = list(range(len(nwbfile.electrodes)))
        electrode_table_region = nwbfile.create_electrode_table_region(
            region=table_ids, description="electrode_table_region"
        )
        dataclass_kwargs = dict(
            electrodes=electrode_table_region,
            conversion=1e-4,
            description=description,
            filtering=info,
        )
        self.add_to_processing_module(
            processing_module=processing_module,
            data=filt_ephys,
            dataclass=ElectricalSeries,
            dataclass_kwargs=dataclass_kwargs,
            containerclass=FilteredEphys,
            container_name="Processed",
            stub_test=stub_test,
        )

        # close redis client
        r.close()

        return nwbfile

    
class StaviskySmoothedSpikingBandPowerInterface(StaviskyTemporalAlignmentInterface):
    """Smoothed spiking band power interface for Stavisky Redis conversion"""
    default_data_kwargs: dict = dict(dtype="float32", encoding="buffer", shape=(1,256))

    def __init__(
        self,
        port: int,
        host: str,
        stream_name: str,
        data_field: str,
        ts_key: str = "spiking_band_power_smoothed",
        frames_per_entry: int = 1,
        data_dtype: Optional[str] = None,
        data_kwargs: dict = dict(),
        nsp_timestamp_field: Optional[str] = "input_nsp_timestamp",
        nsp_timestamp_conversion: Optional[float] = 1.0 / 3.0e4,
        nsp_timestamp_encoding: Optional[str] = "buffer",
        nsp_timestamp_dtype: Optional[Union[str, type, np.dtype]] = "int64",
        nsp_timestamp_index: Optional[int] = 0,
        load_timestamps: bool = True,
        chunk_size: Optional[int] = None,
    ):
        super().__init__(
            port=port, 
            host=host, 
            stream_name=stream_name, 
            data_field=data_field,
            ts_key=ts_key,
            frames_per_entry=frames_per_entry,
            data_dtype=data_dtype,
            data_kwargs=data_kwargs,
            nsp_timestamp_field=nsp_timestamp_field,
            nsp_timestamp_conversion=nsp_timestamp_conversion,
            nsp_timestamp_encoding=nsp_timestamp_encoding,
            nsp_timestamp_dtype=nsp_timestamp_dtype,
            nsp_timestamp_index=nsp_timestamp_index,
            load_timestamps=load_timestamps,
            chunk_size=chunk_size,
        )

    def add_to_nwbfile(
        self,
        nwbfile: NWBFile,
        metadata: Optional[dict] = None,
        stub_test: bool = False,
        chunk_size: Optional[int] = None,
        use_chunk_iterator: bool = False,
        iterator_opts: dict = dict(),
    ):
        # Instantiate Redis client and check connection
        r = redis.Redis(
            port=self.source_data["port"],
            host=self.source_data["host"],
        )
        r.ping()

        # read data
        sbp = self.get_data_iterator(
            client=r, 
            stub_test=stub_test, 
            chunk_size=chunk_size,
            use_chunk_iterator=use_chunk_iterator,
            iterator_opts=iterator_opts,
        )
        
        # get metadata about filtering, etc.
        bin_size = int(self.source_data["stream_name"].split("_")[-1].strip("ms"))
        frequency = 1000 / bin_size
        data = json.loads(r.xrange("supergraph_stream")[0][1][b"data"])
        try:  # shouldn't fail but just in case
            params = data["nodes"]["featureExtraction_and_binning"]["parameters"]
        except Exception as e:
            print(f"Unable to extract filtering info: {e}")
            params = {}
        butter_lowercut = params.get("butter_lowercut", None)
        butter_uppercut = params.get("butter_uppercut", None)
        butter_order = params.get("butter_order", None)
        clip_value = params.get("spike_pow_clip_thresh", None)
        
        try:
            params = data["nodes"]["b2s_preprocess10s"]["parameters"]
        except Exception as e:
            print(f"Unable to extract smoothing info: {e}")
            params = {}
        norm_win_len = params.get("norm_win_len", None)
        smooth_bin_size = params.get("bin_size_ms", None)
        kernel_type = params.get("kernel_type", None)
        kernel_sigma = params.get("kernel_sigma", None)
        kernel_len = params.get("kernel_len", None)

        # build description from metadata
        info = ""
        if butter_lowercut or butter_uppercut:
            info += (
                f"Bandpass filtered with a {butter_order or 'unknown'} order "
                f"Butterworth filter with lower cutoff {butter_lowercut or '0'} Hz "
                f"and upper cutoff {butter_uppercut or 'inf'} Hz."
            )
        if clip_value:
            info += (" " if info else "") + f"Clipped to a maximum of {clip_value}."
        if kernel_type:
            info += (" " if info else "") + (
                f"Smoothed at {smooth_bin_size or 'unknown'} bin size "
                f"using {kernel_type or 'unknown'} kernel type "
                f"with sigma = {kernel_sigma or 'unknown'} "
                f"and kernel length = {kernel_len or 'unknown'}."
            )
        description = f"Spiking band power at {frequency} Hz." + (" " if info else "") + info

        # get processing module
        module_name = "ecephys"
        module_description = "Intermediate data from extracellular electrophysiology recordings, e.g., LFP." 
        processing_module = get_module(nwbfile=nwbfile, name=module_name, description=module_description)
        
        # add to nwbfile
        dataclass_kwargs = dict(
            unit="V^2",
            conversion=1e-8,
            description=description,
        )
        self.add_to_processing_module(
            processing_module=processing_module,
            data=sbp,
            dataclass=TimeSeries,
            dataclass_kwargs=dataclass_kwargs,
            stub_test=stub_test,
        )
        
        # close redis client
        r.close()

        return nwbfile


class StaviskySmoothedThreshCrossingInterface(StaviskyTemporalAlignmentInterface):
    """Smoothed threshold crossing interface for Stavisky conversion"""
    default_data_kwargs: dict = dict(dtype="float32", encoding="buffer", shape=(1,256))

    def __init__(
        self,
        port: int,
        host: str,
        stream_name: str,
        data_field: str,
        ts_key: str = "thresh_crossing_smoothed",
        frames_per_entry: int = 1,
        data_dtype: Optional[str] = None,
        data_kwargs: dict = dict(),
        nsp_timestamp_field: Optional[str] = "input_nsp_timestamp",
        nsp_timestamp_conversion: Optional[float] = 1.0 / 3.0e4,
        nsp_timestamp_encoding: Optional[str] = "buffer",
        nsp_timestamp_dtype: Optional[Union[str, type, np.dtype]] = "int64",
        nsp_timestamp_index: Optional[int] = 0,
        load_timestamps: bool = True,
        chunk_size: Optional[int] = None,
    ):
        super().__init__(
            port=port, 
            host=host, 
            stream_name=stream_name, 
            data_field=data_field,
            ts_key=ts_key,
            frames_per_entry=frames_per_entry,
            data_dtype=data_dtype,
            data_kwargs=data_kwargs,
            nsp_timestamp_field=nsp_timestamp_field,
            nsp_timestamp_conversion=nsp_timestamp_conversion,
            nsp_timestamp_encoding=nsp_timestamp_encoding,
            nsp_timestamp_dtype=nsp_timestamp_dtype,
            nsp_timestamp_index=nsp_timestamp_index,
            load_timestamps=load_timestamps,
            chunk_size=chunk_size,
        )

    def add_to_nwbfile(
        self,
        nwbfile: NWBFile,
        metadata: Optional[dict] = None,
        stub_test: bool = False,
        chunk_size: Optional[int] = None,
        use_chunk_iterator: bool = False,
        iterator_opts: dict = dict(),
    ):
        # Instantiate Redis client and check connection
        r = redis.Redis(
            port=self.source_data["port"],
            host=self.source_data["host"],
        )
        r.ping()
        
        # read data
        thresh_cross = self.get_data_iterator(
            client=r, 
            stub_test=stub_test, 
            chunk_size=chunk_size,
            use_chunk_iterator=use_chunk_iterator,
            iterator_opts=iterator_opts,
        )
        
        data = json.loads(r.xrange("supergraph_stream")[0][1][b"data"])
        try:
            params = data["nodes"]["b2s_preprocess10s"]["parameters"]
        except Exception as e:
            print(f"Unable to extract smoothing info: {e}")
            params = {}
        norm_win_len = params.get("norm_win_len", None)
        smooth_bin_size = params.get("bin_size_ms", None)
        kernel_type = params.get("kernel_type", None)
        kernel_sigma = params.get("kernel_sigma", None)
        kernel_len = params.get("kernel_len", None)

        # build description from metadata
        bin_size = int(self.source_data["stream_name"].split("_")[-1].strip("ms"))
        frequency = 1000 / bin_size
        info = ""
        if kernel_type:
            info += (
                f"Smoothed at {smooth_bin_size or 'unknown'} bin size "
                f"using {kernel_type or 'unknown'} kernel type "
                f"with sigma = {kernel_sigma or 'unknown'} "
                f"and kernel length = {kernel_len or 'unknown'}."
            )
        description = f"Ssmoothed threshold crossings at {frequency} Hz." + (" " if info else "") + info

        # get processing module
        module_name = "ecephys"
        module_description = "Intermediate data from extracellular electrophysiology recordings, e.g., LFP." 
        processing_module = get_module(nwbfile=nwbfile, name=module_name, description=module_description)
        
        # add to nwbfile
        dataclass_kwargs = dict(
            unit="n/a",
            conversion=1.,
            description=description,
        )
        self.add_to_processing_module(
            processing_module=processing_module,
            data=thresh_cross,
            dataclass=TimeSeries,
            dataclass_kwargs=dataclass_kwargs,
            stub_test=stub_test,
        )
        
        # close redis client
        r.close()

        return nwbfile