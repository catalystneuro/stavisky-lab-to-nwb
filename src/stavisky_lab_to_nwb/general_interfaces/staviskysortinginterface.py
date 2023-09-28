"""General sorting interface for Redis stream data."""
import redis
import numpy as np
from pynwb import NWBFile
from typing import Union, Optional, List, Tuple, Literal

from neuroconv.datainterfaces.ecephys.basesortingextractorinterface import BaseSortingExtractorInterface

from .spikeinterface import RedisStreamSortingExtractor
from ..utils.timestamps import get_stream_ids_and_timestamps, smooth_timestamps
from .staviskytemporalalignmentinterface import DualTimestampTemporalAlignmentInterface


class StaviskySortingInterface(BaseSortingExtractorInterface, DualTimestampTemporalAlignmentInterface):
    """Sorting interface for Redis stream data"""

    ExtractorModuleName = "stavisky_lab_to_nwb.general_interfaces.spikeinterface"
    ExtractorName = "RedisStreamSortingExtractor"

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
        timestamp_field: Optional[str] = None,
        timestamp_kwargs: dict = dict(),
        smoothing_kwargs: dict = dict(),
        unit_dim: int = 0,
        clock: Literal["redis", "nsp"] = "nsp",
        chunk_size: int = 10000,
        verbose: bool = True,
    ):
        super().__init__(
            verbose=verbose,
            port=port,
            host=host,
            stream_name=stream_name,
            data_field=data_field,
            data_dtype=data_dtype,
            unit_ids=unit_ids,
            frames_per_entry=frames_per_entry,
            sampling_frequency=sampling_frequency,
            timestamp_field=timestamp_field,
            timestamp_kwargs=timestamp_kwargs,
            smoothing_kwargs=smoothing_kwargs,
            unit_dim=unit_dim,
            clock=clock,
            chunk_size=chunk_size,
        )

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
            return self.sorting_extractor.get_nsp_times()
        else:
            return self.sorting_extractor.get_times()

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
                self.sorting_extractor.set_nsp_times(times=aligned_timestamps)
            else:
                self.sorting_extractor.set_times(times=aligned_timestamps)
        else:
            assert isinstance(
                aligned_timestamps, list
            ), "Recording has multiple segment! Please pass a list of timestamps to align each segment."
            assert (
                len(aligned_timestamps) == self._number_of_segments
            ), f"The number of timestamp vectors ({len(aligned_timestamps)}) does not match the number of segments ({self._number_of_segments})!"

            for segment_index in range(self._number_of_segments):
                if nsp:
                    self.sorting_extractor.set_nsp_times(
                        times=aligned_timestamps[segment_index], segment_index=segment_index
                    )
                else:
                    self.sorting_extractor.set_times(
                        times=aligned_timestamps[segment_index], segment_index=segment_index
                    )

    def get_entry_ids(self):
        return self.sorting_extractor.get_entry_ids()
