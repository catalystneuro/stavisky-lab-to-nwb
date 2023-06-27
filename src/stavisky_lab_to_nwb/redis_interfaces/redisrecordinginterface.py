"""General recording interface for Redis stream data."""
import redis
import numpy as np
from pynwb.file import NWBFile
from typing import Union, Optional, List, Tuple, Literal

from neuroconv.datainterfaces.ecephys.baserecordingextractorinterface import BaseRecordingExtractorInterface

from stavisky_lab_to_nwb.redis_interfaces.redisrecordingextractor import RedisStreamRecordingExtractor


class RedisStreamRecordingInterface(BaseRecordingExtractorInterface):
    """Recording interface for Redis stream data"""

    ExtractorModuleName = "stavisky_lab_to_nwb.redis_interfaces.redisrecordingextractor"
    ExtractorName = "RedisStreamRecordingExtractor"

    def __init__(
        self,
        port: int,
        host: str,
        stream_name: str,
        data_key: str,
        dtype: Union[str, type, np.dtype],
        channel_count: int,
        channel_ids: Optional[list] = None,
        frames_per_entry: int = 1,
        start_time: Optional[float] = None,
        sampling_frequency: Optional[float] = None,
        timestamp_source: Optional[str] = None,
        timestamp_kwargs: dict = {},
        gain_to_uv: Optional[float] = None,
        channel_dim: int = 0,
        verbose: bool = True,
        es_key: str = "ElectricalSeries",
    ):
        super().__init__(
            verbose=verbose,
            es_key=es_key,
            port=port,
            host=host,
            stream_name=stream_name,
            data_key=data_key,
            dtype=dtype,
            channel_count=channel_count,
            channel_ids=channel_ids,
            frames_per_entry=frames_per_entry,
            start_time=start_time,
            sampling_frequency=sampling_frequency,
            timestamp_source=timestamp_source,
            timestamp_kwargs=timestamp_kwargs,
            gain_to_uv=gain_to_uv,
            channel_dim=channel_dim,
        )
