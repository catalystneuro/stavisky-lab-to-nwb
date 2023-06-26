"""Primary class for converting ecephys recording."""
import json
from pynwb import NWBFile

from neuroconv.basedatainterface import BaseDataInterface
from stavisky_lab_to_nwb.redis_interfaces import RedisStreamRecordingInterface


class StaviskyRecordingInterface(RedisStreamRecordingInterface):
    """Recording interface for Stavisky Redis conversion"""

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
    
    def get_metadata(self) -> dict:
        metadata = super().get_metadata()
        
        # connect to Redis
        r = redis.Redis(
            port=self.source_data["port"],
            host=self.source_data["host"],
        )
        r.ping()
        
        # get data if possible
        data = json.loads(r.xrange("supergraph_stream")[0][1][b"data"])
        params = data['nodes']['featureExtraction']['parameters']
        n_arrays = params.get('n_arrays', None)
        n_electrodes_per_array = params.get('n_electrodes_per_array', None)
        r.close()
        
        # Add electrode/device info
        if n_arrays is not None:
            devices = [f"Array{i}" for i in range(1, n_arrays + 1)]
            device_locations = ["unknown", "unknown", "unknown", "unknown"]
            device_metadata = [
                dict(name=device, description=f"Utah array {device}") 
                for device in devices
            ]
            channel_groups = [f"ElectrodeGroup{i}" for i in range(1, n_arrays + 1)]
            electrode_group_metadata = [
                dict(name=str(group_id), description=f"Electrodes from {device}", location=location, device=device)
                for device, location, group_id in zip(devices, device_locations, channel_groups)
            ]
            metadata["Ecephys"].update(dict(
                Device=device_metadata,
                ElectrodeGroup=electrode_group_metadata,
            ))
        
        # TODO: add channel -> electrode group info

        return metadata
