"""General recording interface for Redis stream data."""
import redis
import json
import numpy as np
from pynwb.file import NWBFile
from typing import Union, Optional, List, Tuple, Literal

from neuroconv.datainterfaces.ecephys.baserecordingextractorinterface import BaseRecordingExtractorInterface

from .spikeinterface import RedisStreamRecordingExtractor
from ..utils.timestamps import get_stream_ids_and_timestamps, smooth_timestamps
from .staviskytemporalalignmentinterface import DualTimestampTemporalAlignmentInterface


class StaviskyRecordingInterface(BaseRecordingExtractorInterface, DualTimestampTemporalAlignmentInterface):
    """Recording interface for Redis stream data"""

    ExtractorModuleName = "stavisky_lab_to_nwb.general_interfaces.spikeinterface"
    ExtractorName = "RedisStreamRecordingExtractor"

    def __init__(
        self,
        port: int,
        host: str,
        stream_name: str = "continuousNeural",
        data_field: str = "samples",
        data_dtype: Union[str, type, np.dtype] = "int16",
        channel_ids: Optional[list] = None,
        frames_per_entry: int = 30,
        sampling_frequency: Optional[float] = 3e4,
        timestamp_field: Optional[str] = None,
        timestamp_kwargs: dict = dict(chunk_size=10000),
        smoothing_kwargs: dict = dict(window_len="max", enforce_causal=True),
        gain_to_uv: Optional[float] = 1e-2,
        channel_dim: int = 1,
        chunk_size: int = 10000,
        verbose: bool = True,
        es_key: str = "ElectricalSeries",
    ):
        super().__init__(
            verbose=verbose,
            es_key=es_key,
            port=port,
            host=host,
            stream_name=stream_name,
            data_field=data_field,
            data_dtype=data_dtype,
            channel_ids=channel_ids,
            frames_per_entry=frames_per_entry,
            sampling_frequency=sampling_frequency,
            timestamp_field=timestamp_field,
            timestamp_kwargs=timestamp_kwargs,
            smoothing_kwargs=smoothing_kwargs,
            gain_to_uv=gain_to_uv,
            channel_dim=channel_dim,
            chunk_size=chunk_size,
        )

        # connect to Redis
        r = redis.Redis(
            port=self.source_data["port"],
            host=self.source_data["host"],
        )
        r.ping()

        # get data if possible
        try:
            data = json.loads(r.xrange("supergraph_stream")[0][1][b"data"])
            params = data["nodes"]["featureExtraction_and_binning"]["parameters"]
            self.n_arrays = params.get("n_arrays", None)
            self.n_electrodes_per_array = params.get("n_electrodes_per_array", None)
        except Exception as e:
            print(f"Failed to get recording array metadata: {e}")
            self.n_arrays = None
            self.n_electrodes_per_array = None
        r.close()
        
        if self.n_arrays is not None and self.n_electrodes_per_array is not None:
            n_channels = self.recording_extractor.get_num_channels()
            if n_channels == self.n_arrays * self.n_electrodes_per_array:
                channel_groups = np.repeat(np.arange(self.n_arrays), self.n_electrodes_per_array)
                channel_group_names = [f"Group{i}" for i in (channel_groups + 1)]
                self.recording_extractor.set_channel_groups(channel_group_names)

    def get_metadata(self) -> dict:
        metadata = super().get_metadata()

        # Add electrode/device info
        if self.n_arrays is not None:
            devices = [f"Array{i}" for i in range(1, self.n_arrays + 1)]
            device_locations = ["unknown"] * len(devices)
            device_metadata = [dict(name=device, description=f"Utah array {device}") for device in devices]
            channel_groups = [f"Group{i}" for i in range(1, self.n_arrays + 1)]
            electrode_group_metadata = [
                dict(name=str(group_id), description=f"Electrodes from {device}", location=location, device=device)
                for device, location, group_id in zip(devices, device_locations, channel_groups)
            ]
            metadata["Ecephys"].update(
                dict(
                    Device=device_metadata,
                    ElectrodeGroup=electrode_group_metadata,
                )
            )

        return metadata

    def get_original_timestamps(self):
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
            frames_per_entry=self.source_data.get("frames_per_entry", 1),
            timestamp_field=self.source_data.get("timestamp_field"),
            chunk_size=self.source_data.get("chunk_size", 10000),
            **self.source_data.get("timestamp_kwargs", {})
        )
        if self.source_data.get("smoothing_kwargs", {}):
            redis_timestamps = smooth_timestamps(
                redis_timestamps, 
                frames_per_entry=self.source_data.get("frames_per_entry", 1),
                sampling_frequency=self.source_data.get("sampling_frequency"), 
                **self.source_data.get("smoothing_kwargs"),
            )
        
        # close redis client
        r.close()

        return entry_ids, redis_timestamps, nsp_timestamps

    def get_timestamps(self, nsp: bool = False) -> np.ndarray:
        """
        Retrieve the timestamps for the data in this interface.

        Returns
        -------
        timestamps: numpy.ndarray
            The timestamps for the data stream.
        """
        if nsp:
            return self.recording_extractor.get_nsp_times()
        else:
            return self.recording_extractor.get_times()

    def set_aligned_timestamps(self, aligned_timestamps: np.ndarray, nsp: bool = False) -> None:
        """
        Replace all timestamps for this interface with those aligned to the common session start time.

        Must be in units seconds relative to the common 'session_start_time'.

        Parameters
        ----------
        aligned_timestamps : numpy.ndarray
            The synchronized timestamps for data in this interface.
        """
        # Removed requirement of having recording
        if self._number_of_segments == 1:
            if nsp:
                self.recording_extractor.set_nsp_times(times=aligned_timestamps)
            else:
                self.recording_extractor.set_times(times=aligned_timestamps)
        else:
            assert isinstance(
                aligned_timestamps, list
            ), "Recording has multiple segment! Please pass a list of timestamps to align each segment."
            assert (
                len(aligned_timestamps) == self._number_of_segments
            ), f"The number of timestamp vectors ({len(aligned_timestamps)}) does not match the number of segments ({self._number_of_segments})!"

            for segment_index in range(self._number_of_segments):
                if nsp:
                    self.recording_extractor.set_nsp_times(
                        times=aligned_timestamps[segment_index], segment_index=segment_index)
                else:
                    self.recording_extractor.set_times(
                        times=aligned_timestamps[segment_index], segment_index=segment_index)
    
    def get_entry_ids(self):
        return self.recording_extractor.get_entry_ids()