"""Primary script to run to convert an entire session for of data using the NWBConverter."""
from pathlib import Path
from typing import Union
import datetime
from zoneinfo import ZoneInfo
import redis
import numpy as np

from neuroconv.utils import load_dict_from_file, dict_deep_update

from stavisky_lab_to_nwb.simulated_data import SimulatedDataNWBConverter


def session_to_nwb(port: int, host: str, output_dir_path: Union[str, Path], stub_test: bool = False):

    r = redis.Redis(port=port, host=host)
    r.ping()
    rdb_metadata = r.xrange('metadata')[0][1]

    output_dir_path = Path(output_dir_path)
    if stub_test:
        output_dir_path = output_dir_path / "nwb_stub"
    output_dir_path.mkdir(parents=True, exist_ok=True)
    
    session_id = str(rdb_metadata[b'session_name'])
    nwbfile_path = output_dir_path / f"{session_id}.nwb"

    source_data = dict()
    conversion_options = dict()

    # Add Recording
    source_data.update(dict(Recording=dict(port=port, host=host)))
    conversion_options.update(dict(Recording=dict()))
    
    # Add LFP
    # source_data.update(dict(LFP=dict()))
    # conversion_options.update(dict(LFP=dict()))

    # Add Sorting
    # source_data.update(dict(Sorting=dict()))
    # conversion_options.update(dict(Sorting=dict()))

    # Add Behavior
    # source_data.update(dict(Behavior=dict()))
    # conversion_options.update(dict(Behavior=dict()))
    
    # Add Trials
    source_data.update(dict(Trials=dict(port=port, host=host)))
    conversion_options.update(dict(Trials=dict()))

    converter = SimulatedDataNWBConverter(source_data=source_data)
    
    # Add datetime to conversion
    metadata = converter.get_metadata()
    date = datetime.datetime.fromtimestamp(np.frombuffer(
            rdb_metadata[b'startTime'],
            dtype=np.float64).item()).replace(tzinfo=ZoneInfo("US/Pacific"))
    metadata["NWBFile"]["session_start_time"] = date
    
    # Add subject ID
    subject = str(rdb_metadata[b'participant'])
    metadata["Subject"]["subject_id"] = subject
    
    # Add session info
    metadata["NWBFile"]["session_id"] = session_id
    session_description = f"{str(rdb_metadata[b'session_description']).strip('. \n')}. " + \
        f"Block {str(rdb_metadata[b'block_num']).strip()}: {str(rdb_metadata[b'block_description']).strip()}"
    metadata["NWBFile"]["session_description"] = session_description
    
    # Close redis instance
    r.close()
    
    # Update default metadata with the editable in the corresponding yaml file
    editable_metadata_path = Path(__file__).parent / "simulated_data_metadata.yaml"
    editable_metadata = load_dict_from_file(editable_metadata_path)
    metadata = dict_deep_update(metadata, editable_metadata)

    # Run conversion
    converter.run_conversion(metadata=metadata, nwbfile_path=nwbfile_path, conversion_options=conversion_options)
    

if __name__ == "__main__":
    
    # Parameters for conversion
    port = 6379
    host = "localhost"
    output_dir_path = Path("~/conversion_nwb/stavisky-lab-to-nwb/simulated_data/").expanduser()
    stub_test = False

    session_to_nwb(port=port, 
                    host=host,
                    output_dir_path=output_dir_path, 
                    stub_test=stub_test,
                    )