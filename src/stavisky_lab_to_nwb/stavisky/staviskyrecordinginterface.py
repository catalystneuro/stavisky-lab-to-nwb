"""Primary class for converting ecephys recording."""
from pynwb.file import NWBFile

from neuroconv.basedatainterface import BaseDataInterface


class RedisRecordingInterface(BaseDataInterface):
    """Recording interface for Stavisky Redis conversion"""

    def __init__(self):
        # This should load the data lazily and prepare variables you need
        pass

    def get_metadata(self):
        # Automatically retrieve as much metadata as possible
        metadata = super().get_metadata()

        return metadata

    def run_conversion(self, nwbfile: NWBFile, metadata: dict):
        # All the custom code to write to PyNWB

        return nwbfile
