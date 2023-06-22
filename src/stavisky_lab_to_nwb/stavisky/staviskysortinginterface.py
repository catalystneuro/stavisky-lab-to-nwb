"""Primary class for converting sorting data."""
from pynwb.file import NWBFile

from neuroconv.basedatainterface import BaseDataInterface


<<<<<<< HEAD:src/stavisky_lab_to_nwb/stavisky/staviskysortinginterface.py
class RedisSortingInterface(BaseDataInterface):
    """Sorting interface for Stavisky Redis conversion"""
=======
class SimulatedDataBehaviorInterface(BaseDataInterface):
    """Behavior interface for simulated_data conversion"""
>>>>>>> main:src/stavisky_lab_to_nwb/simulated_data/simulated_databehaviorinterface.py

    def __init__(self):
        # This should load the data lazily and prepare variables you need
        pass

    def get_metadata(self):
        # Automatically retrieve as much metadata as possible
        metadata = super().get_metadata()

        return metadata

    def add_to_nwbfile(self, nwbfile: NWBFile, metadata: dict):
        # All the custom code to write to PyNWB

        return nwbfile
