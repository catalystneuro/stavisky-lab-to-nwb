from pynwb import NWBFile, TimeSeries
from ndx_sound import AcousticWaveformSeries
from typing import Optional, Union
import json
import numpy as np
import redis

from ..general_interfaces import StaviskyTemporalAlignmentInterface


class BrainToTextAudioInterface(StaviskyTemporalAlignmentInterface):
    default_data_kwargs: dict = dict(dtype="int16", encoding="buffer", shape=(30, 2))

    def __init__(  # TODO: smooth timestamps somehow
        self,
        port: int,
        host: str,
        stream_name: str,
        data_field: str,
        ts_key: str = "audio",
        frames_per_entry: int = 30,
        data_dtype: Optional[str] = None,
        data_kwargs: dict = dict(),
        nsp_timestamp_field: Optional[str] = "timestamps",
        nsp_timestamp_conversion: Optional[float] = 1.0e-9,
        nsp_timestamp_encoding: Optional[str] = "buffer",
        nsp_timestamp_dtype: Optional[Union[str, type, np.dtype]] = "int64",
        nsp_timestamp_index: Optional[int] = None,
        smoothing_kwargs: dict = dict(window_len="max", sampling_frequency=3.0e4),
        load_timestamps: bool = True,
        chunk_size: Optional[int] = None,
    ):
        super().__init__(
            port=port,
            host=host,
            stream_name=stream_name,
            data_field=data_field,
            ts_key=ts_key,
            data_dtype=data_dtype,
            frames_per_entry=frames_per_entry,
            nsp_timestamp_field=nsp_timestamp_field,
            nsp_timestamp_conversion=nsp_timestamp_conversion,
            nsp_timestamp_encoding=nsp_timestamp_encoding,
            nsp_timestamp_dtype=nsp_timestamp_dtype,
            nsp_timestamp_index=nsp_timestamp_index,
            smoothing_kwargs=smoothing_kwargs,
            load_timestamps=load_timestamps,
            chunk_size=chunk_size,
        )

    def add_to_nwbfile(
        self,
        nwbfile: NWBFile,
        metadata: Optional[dict] = None,
        stub_test: bool = False,
        chunk_size: Optional[int] = None,
    ):
        # Instantiate Redis client and check connection
        r = redis.Redis(
            port=self.source_data["port"],
            host=self.source_data["host"],
        )
        r.ping()

        # read data
        stream_name = self.source_data["stream_name"]
        data_field = self.source_data["data_field"]
        analog = self.get_data_iterator(client=r, stub_test=stub_test, chunk_size=chunk_size)
        assert len(np.unique(analog[:, 1])) == 1
        analog = analog[:, 0]  # drop second channel as it has only one value

        # stub timestamps if necessary
        timestamps = self.get_timestamps(nsp=False)
        if stub_test:
            timestamps = timestamps[: len(analog)]
        assert len(timestamps) == len(analog), "Timestamps and data have different lengths!"

        # get metadata about filtering, etc.
        data = json.loads(r.xrange("supergraph_stream")[0][1][b"data"])
        try:  # shouldn't fail but just in case
            params = data["nodes"]["cerebusAdapter_mic"]["parameters"]
        except Exception as e:
            print(f"Unable to extract filtering info: {e}")
            params = {}
        sampling_freq = params.get("samp_freq", [None])[0]
        if sampling_freq is not None:
            sampling_freq = float(sampling_freq)

        # check sampling frequency
        if sampling_freq is not None:
            timestamps_freq = 1.0 / np.mean(np.diff(timestamps))
            assert np.isclose(
                sampling_freq, timestamps_freq
            ), f"Data sampling frequency differs from stated value. {timestamps_freq} != {sampling_freq}"

        # build description from metadata
        description = f"Raw microphone audio recorded at {sampling_freq or 'unknown'} Hz."

        # add to nwbfile
        dataclass = AcousticWaveformSeries
        dataclass_kwargs = dict(
            starting_time=timestamps[0],
            rate=(1.0 / np.mean(np.diff(timestamps))),
            description=description,
        )

        data_interface = dataclass(
            data=analog,
            name=self.ts_key,
            **dataclass_kwargs,
        )
        nwbfile.add_acquisition(data_interface)

        # close redis client
        r.close()

        return nwbfile
