"""Primary script to run to convert an entire session for of data using the NWBConverter."""

from pathlib import Path
from typing import Union
import datetime
from zoneinfo import ZoneInfo
import redis
import numpy as np

from neuroconv.utils import load_dict_from_file, dict_deep_update

from stavisky_lab_to_nwb.braintotext import BrainToTextNWBConverter


def session_to_nwb(
    port: int,
    host: str,
    conversion_config_path: Union[str, Path],
    output_dir_path: Union[str, Path],
    source_data: dict = dict(),
    conversion_options: dict = dict(),
    stub_test: bool = False,
    exclude_interfaces: list[str] = [],
):
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

    converter = BrainToTextNWBConverter(
        conversion_config_path=conversion_config_path,
        port=port,
        host=host,
        source_data=source_data,
        session_start_time=session_start_time,
        exclude_interfaces=exclude_interfaces,
    )

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
    converter.run_conversion(
        metadata=metadata,
        nwbfile_path=nwbfile_path,
        conversion_options=conversion_options,
        stub_test=stub_test,
    )


if __name__ == "__main__":
    # Parameters for conversion
    port = 6379
    host = "localhost"
    conversion_config_path = Path(__file__).parent / Path("braintotext_conversion.yaml")
    output_dir_path = Path("~/conversion_nwb/stavisky-lab-to-nwb/braintotext/").expanduser()
    stub_test = True

    session_to_nwb(
        port=port,
        host=host,
        conversion_config_path=conversion_config_path,
        output_dir_path=output_dir_path,
        stub_test=stub_test,
    )
