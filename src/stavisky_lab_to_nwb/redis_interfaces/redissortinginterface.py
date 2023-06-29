"""General sorting interface for Redis stream data."""
import redis
import numpy as np
from pynwb import NWBFile
from typing import Union, Optional, List, Tuple, Literal

from neuroconv.datainterfaces.ecephys.basesortingextractorinterface import BaseSortingExtractorInterface

from stavisky_lab_to_nwb.redis_interfaces.redissortingextractor import RedisStreamSortingExtractor


class RedisStreamSortingInterface(BaseSortingExtractorInterface):
    """Sorting interface for Redis stream data"""

    ExtractorModuleName = "stavisky_lab_to_nwb.redis_interfaces.redissortingextractor"
    ExtractorName = "RedisStreamSortingExtractor"

    def __init__(
        self,
        port: int,
        host: str,
        stream_name: str,
        data_key: str,
        dtype: Union[str, type, np.dtype],
        unit_count: int,
        unit_ids: Optional[list] = None,
        frames_per_entry: int = 1,
        start_time: Optional[float] = None,
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
            unit_count=unit_count,
            unit_ids=unit_ids,
            frames_per_entry=frames_per_entry,
            start_time=start_time,
            sampling_frequency=sampling_frequency,
            timestamp_source=timestamp_source,
            timestamp_kwargs=timestamp_kwargs,
            unit_dim=unit_dim,
        )
