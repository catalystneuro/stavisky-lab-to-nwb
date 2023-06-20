"""Primary class for converting sorting data."""
import redis
import numpy as np
from pynwb import NWBFile
from typing import Union, Optional, List, Tuple, Literal

from neuroconv.utils import DeepDict
from neuroconv.datainterfaces.ecephys.basesortingextractorinterface import BaseSortingExtractorInterface

from stavisky_lab_to_nwb.redis_interfaces.redissortingextractor import RedisStreamSortingExtractor


class RedisSortingInterface(BaseSortingExtractorInterface):
    """Sorting interface for Stavisky Redis conversion"""
    
    ExtractorModuleName = "stavisky_lab_to_nwb.redis_interfaces.redissortingextractor"
    ExtractorName = "RedisStreamSortingExtractor"
    
    def __init__(
        self,
        port: int,
        host: str,
        stream_name: str,
        data_key: str,
        unit_count: int,
        dtype: Union[str, type, np.dtype],
        unit_ids: Optional[list] = None,
        frames_per_entry: int = 1,
        timestamps: Optional[list] = None,
        start_time: Optional[float] = None,
        sampling_frequency: Optional[float] = None,
        timestamp_source: str = "redis",
        timestamp_kwargs: dict = {},
        recording_frequency_ratio: int = 1,
        unit_dim: int = 0,
        verbose: bool = True,
    ):
        super().__init__(
            verbose=verbose,
            port=port,
            host=host,
            stream_name=stream_name,
            data_key=data_key,
            unit_count=unit_count,
            dtype=dtype,
            unit_ids=unit_ids,
            frames_per_entry=frames_per_entry,
            start_time=start_time,
            sampling_frequency=sampling_frequency,
            timestamp_source=timestamp_source,
            timestamp_kwargs=timestamp_kwargs,
            unit_dim=unit_dim,
            recording_frequency_ratio=recording_frequency_ratio,
        )

    def get_metadata(self):
        # Automatically retrieve as much metadata as possible
        metadata = super().get_metadata()
        
        return metadata

        
if __name__ == "__main__":
    extractor = RedisStreamSortingExtractor(
        port=6379,
        host="localhost",
        stream_name="neuralFeatures_1ms",
        data_key=b'threshold_crossings',
        unit_count=256,
        dtype=np.int16,
        unit_ids=None,
        frames_per_entry=1,
        start_time=None,
        sampling_frequency=None,
        timestamps=None,
        timestamp_source="redis",
        timestamp_kwargs={
            "chunk_size": 10000,
            "timestamp_unit": "ms",
            "smoothing_window": 1,
            "smoothing_stride": 1,
        },
    )
    out = extractor.get_all_spike_trains(
        outputs="unit_index",
    )
    times, labels = out[0]
    import pdb; pdb.set_trace()