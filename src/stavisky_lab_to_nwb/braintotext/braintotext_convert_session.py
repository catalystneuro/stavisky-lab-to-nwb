"""Primary script to run to convert an entire session for of data using the NWBConverter."""
from pathlib import Path
from typing import Union
import datetime
from zoneinfo import ZoneInfo
import redis
import numpy as np

from neuroconv.utils import load_dict_from_file, dict_deep_update

from stavisky_lab_to_nwb.braintotext import BrainToTextNWBConverter


def session_to_nwb(port: int, host: str, output_dir_path: Union[str, Path], stub_test: bool = False):
    # Instantiate Redis client and check connection
    r = redis.Redis(port=port, host=host)
    r.ping()

    # Extract session metadata
    rdb_metadata = r.xrange("metadata")[0][1]
    session_start_time = np.frombuffer(rdb_metadata[b"startTime"], dtype=np.int64).astype(float).item()

    # Prepare output path
    output_dir_path = Path(output_dir_path)
    if stub_test:
        output_dir_path = output_dir_path / "nwb_stub"
    output_dir_path.mkdir(parents=True, exist_ok=True)

    session_id = rdb_metadata[b"session_name"].decode("UTF-8")
    nwbfile_path = output_dir_path / f"{session_id}.nwb"

    # Configure conversion
    source_data = dict()
    conversion_options = dict()

    # Add Recording
    source_data.update(
        dict(
            Recording=dict(
                port=port,
                host=host,
                stream_name="continuousNeural",
                data_field="samples",
                data_dtype="int16",
                frames_per_entry=30,
                sampling_frequency=3.0e4,
                timestamp_field="timestamps",
                timestamp_kwargs=dict(
                    timestamp_dtype="int64",
                    timestamp_encoding="buffer",
                    timestamp_conversion=1.0 / 3e4,
                ),
                smoothing_kwargs=dict(
                    window_len="max",
                    enforce_causal=True,
                ),
                gain_to_uv=0.01,
                channel_dim=1,
            )
        )
    )
    conversion_options.update(
        dict(
            Recording=dict(
                iterator_opts=dict(
                    buffer_gb=0.2,  # may need to reduce depending on machine
                ),
                stub_test=stub_test,
            )
        )
    )

    # Add Sorting
    source_data.update(
        dict(
            Sorting=dict(
                port=port,
                host=host,
                stream_name="binnedFeatures_10ms",
                data_field="threshold_crossings",
                data_dtype="int16",
                frames_per_entry=1,
                timestamp_field="input_nsp_timestamp",
                timestamp_kwargs=dict(
                    timestamp_conversion=1.0 / 3.0e4,
                    timestamp_encoding="buffer",
                    timestamp_dtype="int64",
                    timestamp_index=0,
                ),
                chunk_size=50000,
                clock="nsp",
            )
        )
    )
    conversion_options.update(
        dict(
            Sorting=dict(
                units_description=(
                    "Unsorted threshold crossings binned at 10 ms resolution for each recording channel."
                ),
                write_as="processing",
                stub_test=stub_test,
            )
        )
    )

    # Add SpikingBandPower 1 ms resolution
    # source_data.update(
    #     dict(
    #         SpikingBandPower1ms=dict(
    #             port=port,
    #             host=host,
    #             stream_name="neuralFeatures_1ms",
    #             data_key="spike_band_power",
    #             ts_key="spiking_band_power_1ms",
    #         )
    #     )
    # )
    # conversion_options.update(
    #     dict(
    #         SpikingBandPower1ms=dict(
    #             stub_test=stub_test,
    #             smooth_timestamps=False,
    #             chunk_size=10000,
    #         )
    #     )
    # )

    # Add SpikingBandPower 20 ms resolution
    # source_data.update(
    #     dict(
    #         SpikingBandPower20ms=dict(
    #             port=port,
    #             host=host,
    #             stream_name="binnedFeatures_20ms",
    #             data_key="spike_band_power_bin",
    #             ts_key="spiking_band_power_20ms",
    #         )
    #     )
    # )
    # conversion_options.update(
    #     dict(
    #         SpikingBandPower20ms=dict(
    #             stub_test=stub_test,
    #             smooth_timestamps=False,
    #             chunk_size=1000,
    #         )
    #     )
    # )

    # Add audio
    source_data.update(
        dict(
            Audio=dict(
                port=port,
                host=host,
                stream_name="continuousAnalog",
                data_field="samples",
                ts_key="audio",
                nsp_timestamp_field="timestamps",
                nsp_timestamp_conversion=1.0e-9,
                nsp_timestamp_encoding="buffer",
                nsp_timestamp_dtype="int64",
                smoothing_kwargs=dict(window_len="max", sampling_frequency=3.0e4),
                chunk_size=10000,
            )
        )
    )
    conversion_options.update(
        dict(
            Audio=dict(
                stub_test=stub_test,
            )
        )
    )

    # Add Trials
    source_data.update(dict(Trials=dict(port=port, host=host)))
    conversion_options.update(dict(Trials=dict()))

    # Add Decoding
    source_data.update(dict(PhonemeLogits=dict(port=port, host=host)))
    conversion_options.update(dict(PhonemeLogits=dict()))
    source_data.update(dict(DecodedText=dict(port=port, host=host)))
    conversion_options.update(dict(DecodedText=dict()))

    converter = BrainToTextNWBConverter(source_data=source_data, session_start_time=session_start_time)

    # Add datetime to conversion
    metadata = converter.get_metadata()
    date = datetime.datetime.fromtimestamp(session_start_time).astimezone(tz=ZoneInfo("US/Pacific"))
    metadata["NWBFile"]["session_start_time"] = date

    # Add subject ID
    subject = rdb_metadata[b"participant"].decode("UTF-8")
    metadata["Subject"]["subject_id"] = subject

    # Add session info
    metadata["NWBFile"]["session_id"] = session_id
    session_description = (
        f"{rdb_metadata[b'session_description'].decode('UTF-8').strip()}. "
        + f"Block {rdb_metadata[b'block_num'].decode('UTF-8').strip()}: "
        + f"{rdb_metadata[b'block_description'].decode('UTF-8').strip()}"
    )
    metadata["NWBFile"]["session_description"] = session_description

    # Close Redis client
    r.close()

    # Update default metadata with the editable in the corresponding yaml file
    editable_metadata_path = Path(__file__).parent / "braintotext_metadata.yaml"
    editable_metadata = load_dict_from_file(editable_metadata_path)
    metadata = dict_deep_update(metadata, editable_metadata)

    # Run conversion
    converter.run_conversion(metadata=metadata, nwbfile_path=nwbfile_path, conversion_options=conversion_options)


if __name__ == "__main__":
    # Parameters for conversion
    port = 6379
    host = "localhost"
    output_dir_path = Path("~/conversion_nwb/stavisky-lab-to-nwb/braintotext/").expanduser()
    stub_test = True

    session_to_nwb(
        port=port,
        host=host,
        output_dir_path=output_dir_path,
        stub_test=stub_test,
    )
