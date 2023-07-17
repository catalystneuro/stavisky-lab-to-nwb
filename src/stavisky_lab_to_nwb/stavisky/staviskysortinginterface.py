"""Primary class for converting sorting data."""
import json
import numpy as np
import redis
from pynwb import NWBFile
from typing import Union, Optional, Literal

from neuroconv.utils import DeepDict
from stavisky_lab_to_nwb.redis_interfaces import RedisStreamSortingInterface


class StaviskySortingInterface(RedisStreamSortingInterface):
    """Sorting interface for Stavisky Redis conversion"""

    def __init__(
        self,
        port: int,
        host: str,
        stream_name: str,
        data_key: str,
        dtype: Union[str, type, np.dtype],
        unit_ids: Optional[list] = None,
        frames_per_entry: int = 1,
        sampling_frequency: Optional[float] = None,
        timestamp_source: Optional[str] = None,
        timestamp_kwargs: dict = {},
        unit_dim: int = 0,
        verbose: bool = True,
    ):
        super().__init__(
            verbose=verbose,
            port=port,
            host=host,
            stream_name=stream_name,
            data_key=data_key,
            dtype=dtype,
            unit_ids=unit_ids,
            frames_per_entry=frames_per_entry,
            sampling_frequency=sampling_frequency,
            timestamp_source=timestamp_source,
            timestamp_kwargs=timestamp_kwargs,
            unit_dim=unit_dim,
        )

    def get_original_timestamps(self) -> np.ndarray:
        """
        Retrieve the original unaltered timestamps for the data in this interface.

        This function should retrieve the data on-demand by re-initializing the IO.

        Returns
        -------
        timestamps: numpy.ndarray
            The timestamps for the data stream.
        """
        return self.sorting_extractor.get_ids_and_timestamps(
            stream_name=self.source_data.get("stream_name"),
            frames_per_entry=self.source_data.get("frames_per_entry", 1),
            sampling_frequency=self.source_data.get("sampling_frequency", None),
            timestamp_source=self.source_data.get("timestamp_source", None),
            **self.source_data.get("timestamp_kwargs", dict()),
        )[1]

    def get_timestamps(self) -> np.ndarray:
        """
        Retrieve the timestamps for the data in this interface.

        Returns
        -------
        timestamps: numpy.ndarray
            The timestamps for the data stream.
        """
        return self.sorting_extractor.get_times()

    def set_aligned_timestamps(self, aligned_timestamps: np.ndarray) -> None:
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
            self.sorting_extractor.set_times(times=aligned_timestamps)
        else:
            assert isinstance(
                aligned_timestamps, list
            ), "Recording has multiple segment! Please pass a list of timestamps to align each segment."
            assert (
                len(aligned_timestamps) == self._number_of_segments
            ), f"The number of timestamp vectors ({len(aligned_timestamps)}) does not match the number of segments ({self._number_of_segments})!"

            for segment_index in range(self._number_of_segments):
                self.sorting_extractor.set_times(times=aligned_timestamps[segment_index], segment_index=segment_index)

    def set_aligned_starting_time(self, aligned_starting_time: float) -> None:
        """
        Align the starting time for this interface relative to the common session start time.

        Must be in units seconds relative to the common 'session_start_time'.

        Parameters
        ----------
        aligned_starting_time : float
            The starting time for all temporal data in this interface.
        """
        self.set_aligned_timestamps(aligned_timestamps=self.get_timestamps() + aligned_starting_time)
