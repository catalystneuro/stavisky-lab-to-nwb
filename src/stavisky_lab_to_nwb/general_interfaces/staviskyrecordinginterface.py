"""General recording interface for Redis stream data."""

import redis
import json
import numpy as np
from pathlib import Path
from pynwb.file import NWBFile
from typing import Union, Optional, List, Tuple, Literal

from neuroconv.datainterfaces.ecephys.baserecordingextractorinterface import BaseRecordingExtractorInterface
from neuroconv.utils.types import FilePathType

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
        nsp_timestamp_field: Optional[str] = None,
        nsp_timestamp_kwargs: dict = dict(),
        smoothing_kwargs: dict = dict(window_len="max", enforce_causal=True),
        gain_to_uv: Optional[float] = 1e-2,
        channel_dim: int = 1,
        buffer_gb: Optional[float] = None,
        verbose: bool = True,
        es_key: str = "ElectricalSeries",
        channel_mapping_file: Optional[FilePathType] = None,
    ):
        """Initialize the StaviskyRecordingInterface

        Parameters
        ----------
        port : int
            Port number for Redis server.
        host : str
            Host name for Redis server, e.g. "localhost".
        stream_name : str
            Name of stream containing the recording data.
        data_field : bytes or str
            Key or field within each Redis stream entry that
            contains the recording data.
        data_dtype : str, type, or numpy.dtype
            The dtype of the data. Assumed to be a numeric type
            recognized by numpy, e.g. int8, float32, etc.
        channel_ids : list, optional
            List of ids for each channel. If not provided, the channel
            indices are used as ids.
        frames_per_entry : int, default: 1
            Number of frames (i.e. single time points/samples) contained
            within each Redis stream entry.
        sampling_frequency : float, optional
            The sampling frequency of the data in Hz. If not provided,
            it is inferred from the timestamps.
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
        gain_to_uv : float, optional
            Scaling necessary to convert the recording values to
            microvolts.
        channel_dim : int, default: 0
            If frames_per_entry > 1, then channel_dim indicates the
            axis ordering of the data in each entry. If channel_dim
            = 0, then the data are assumed to originally be of shape
            (channel_count, frames_per_entry). If channel_dim = 1,
            then the data are assumed to originally be of shape
            (frames_per_entry, channel_count)
        buffer_gb : float, optional
            The amount of data to read simultaneously when
            iterating through the Redis stream, in gb. If `None`, the
            entire stream will be read from Redis at once
        verbose : bool, default: True
            If True, will print out additional information.
        es_key : str, default: "ElectricalSeries"
            The key of this ElectricalSeries when saved to NWB
        channel_mapping_file: Path or str, optional
            Path to JSON file specifying channel mapping to unscramble
            continuous data channels. Provided path can be either
            absolute, or relative to `src/stavisky_lab_to_nwb/`
        """
        if channel_mapping_file is not None:
            channel_mapping_file = Path(channel_mapping_file)
            if not channel_mapping_file.is_absolute():
                channel_mapping_file = Path(__file__).parent.parent / channel_mapping_file
            with open(channel_mapping_file) as f:
                channel_mapping = (np.asarray(json.load(f)["electrode_mapping"], dtype=int) - 1).tolist()
        else:
            channel_mapping = None
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
            nsp_timestamp_field=nsp_timestamp_field,
            nsp_timestamp_kwargs=nsp_timestamp_kwargs,
            smoothing_kwargs=smoothing_kwargs,
            gain_to_uv=gain_to_uv,
            channel_dim=channel_dim,
            buffer_gb=buffer_gb,
            channel_mapping=channel_mapping,
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
            buffer_gb=self.source_data.get("buffer_gb", None),
            **self.source_data.get("timestamp_kwargs", {}),
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
                        times=aligned_timestamps[segment_index], segment_index=segment_index
                    )
                else:
                    self.recording_extractor.set_times(
                        times=aligned_timestamps[segment_index], segment_index=segment_index
                    )

    def get_entry_ids(self):
        return self.recording_extractor.get_entry_ids()

    def add_to_nwbfile(
        self,
        nwbfile: NWBFile,
        metadata: Optional[dict] = None,
        stub_test: bool = False,
        starting_time: Optional[float] = None,
        write_as: Literal["raw", "lfp", "processed"] = "raw",
        write_electrical_series: bool = True,
        compression: Optional[str] = "gzip",
        compression_opts: Optional[int] = None,
        iterator_type: str = "v2",
        iterator_opts: Optional[dict] = None,
    ):
        if iterator_type == "v2" and "buffer_gb" not in iterator_opts:
            iterator_opts["buffer_gb"] = self.source_data.get("buffer_gb", None)
        super().add_to_nwbfile(
            nwbfile=nwbfile,
            metadata=metadata,
            stub_test=stub_test,
            starting_time=starting_time,
            write_as=write_as,
            write_electrical_series=write_electrical_series,
            compression=compression,
            compression_opts=compression_opts,
            iterator_type=iterator_type,
            iterator_opts=iterator_opts,
        )
