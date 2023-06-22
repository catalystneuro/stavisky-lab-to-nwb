"""Primary class for converting ecephys recording."""
import redis
import numpy as np
from pynwb.file import NWBFile
from typing import Union, Optional, List, Tuple, Literal

from neuroconv.datainterfaces.ecephys.baserecordingextractorinterface import BaseRecordingExtractorInterface

from stavisky_lab_to_nwb.redis_interfaces.redisrecordingextractor import RedisStreamRecordingExtractor


class RedisRecordingInterface(BaseRecordingExtractorInterface):
    """Recording interface for Stavisky Redis conversion"""
    
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


if __name__ == "__main__":
    extractor = RedisStreamRecordingExtractor(
        port=6379,
        host="localhost",
        stream_name="continuousNeural",
        data_key=b'samples',
        channel_count=256,
        dtype=np.int16,
        channel_ids=None,
        frames_per_entry=30,
        start_time=None,
        sampling_frequency=None,
        timestamps=None,
        timestamp_source=b'BRANDS_time',
        timestamp_kwargs={
            "chunk_size": 10000,
            "timestamp_unit": "s",
            "smoothing_window": 30,
            "smoothing_stride": 30,
            "timestamp_dtype": np.float64,
            "timestamp_encoding": "buffer",
        },
        gain_to_uv=100.,
        channel_dim=1,
    )
    traces = extractor.get_traces(
        segment_index=None,
        start_frame=0,
        end_frame=30000,
        channel_ids=None,
        order=None,
        return_scaled=False,
        cast_unsigned=False,
    )
    import pdb; pdb.set_trace()
