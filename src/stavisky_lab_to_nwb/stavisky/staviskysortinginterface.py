"""Primary class for converting sorting data."""
from pynwb.file import NWBFile

from neuroconv.basedatainterface import BaseDataInterface

class RedisSortingInterface(BaseDataInterface):
    """Sorting interface for Stavisky Redis conversion"""

    def __init__(self):
        # This should load the data lazily and prepare variables you need

    def get_metadata(self):
        # Automatically retrieve as much metadata as possible
        metadata = super().get_metadata()

        return metadata

    def add_to_nwbfile(self, nwbfile: NWBFile, metadata: dict):
        # All the custom code to write to PyNWB

        return nwbfile
