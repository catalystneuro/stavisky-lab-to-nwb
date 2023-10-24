from pynwb import NWBFile, TimeSeries
from ndx_sound import AcousticWaveformSeries
from typing import Optional, Union
from hdmf.backends.hdf5 import H5DataIO
import json
import numpy as np
import redis

from .staviskytemporalalignmentinterface import StaviskyTemporalAlignmentInterface


class StaviskyAudioInterface(StaviskyTemporalAlignmentInterface):
    default_data_kwargs: dict = dict(dtype="int16", encoding="buffer", shape=(30, 2))

    def __init__(
        self,
        port: int,
        host: str,
        stream_name: str,
        data_field: str,
        ts_key: str = "AudioSeries",
        frames_per_entry: int = 30,
        data_dtype: Optional[str] = None,
        data_kwargs: dict = dict(),
        nsp_timestamp_field: Optional[str] = "timestamps",
        nsp_timestamp_kwargs: dict = dict(),
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
            nsp_timestamp_kwargs=nsp_timestamp_kwargs,
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
        use_chunk_iterator: bool = False,
        iterator_opts: dict = dict(),
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
        analog = self.get_data_iterator(
            client=r,
            stub_test=stub_test,
            chunk_size=chunk_size,
            use_chunk_iterator=use_chunk_iterator,
            iterator_opts=iterator_opts,
        )
        single_value_columns = np.array([(len(np.unique(analog[:, i])) == 1) for i in range(analog.shape[-1])])
        if np.any(single_value_columns):
            print(f"StaviskyAudioInterface: Dropping columns {np.nonzero(single_value_columns)[0]} as they have only one unique value")
            analog = analog[:, ~single_value_columns]
            if analog.shape[-1] == 0:
                print(f"StaviskyAudioInterface: All data dropped. Skipping this data stream...")
                return

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
            print(f"StaviskyAudioInterface: Unable to extract filtering info: {e}")
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
            data=H5DataIO(analog, compression="gzip"),
            name=self.ts_key,
            **dataclass_kwargs,
        )
        nwbfile.add_acquisition(data_interface)

        # close redis client
        r.close()

        return nwbfile
