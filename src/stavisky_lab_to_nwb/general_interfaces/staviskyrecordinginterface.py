"""Primary class for converting ecephys recording."""
import json
import numpy as np
import redis
from pynwb import NWBFile
from typing import Union, Optional, Literal

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
        channel_ids: Optional[list] = None,
        frames_per_entry: int = 1,
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
            channel_ids=channel_ids,
            frames_per_entry=frames_per_entry,
            sampling_frequency=sampling_frequency,
            timestamp_source=timestamp_source,
            timestamp_kwargs=timestamp_kwargs,
            gain_to_uv=gain_to_uv,
            channel_dim=channel_dim,
        )

        # connect to Redis
        r = redis.Redis(
            port=self.source_data["port"],
            host=self.source_data["host"],
        )
        r.ping()

        # get data if possible
        data = json.loads(r.xrange("supergraph_stream")[0][1][b"data"])
        params = data["nodes"]["featureExtraction"]["parameters"]
        self.n_arrays = params.get("n_arrays", None)
        self.n_electrodes_per_array = params.get("n_electrodes_per_array", None)
        r.close()

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
        n_channels = self.recording_extractor.get_num_channels()
        if n_channels == self.n_arrays * self.n_electrodes_per_array:
            channel_groups = np.repeat(np.arange(self.n_arrays), self.n_electrodes_per_array)
            channel_group_names = [f"Group{i}" for i in (channel_groups + 1)]
            self.recording_extractor.set_channel_groups(channel_group_names)

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
