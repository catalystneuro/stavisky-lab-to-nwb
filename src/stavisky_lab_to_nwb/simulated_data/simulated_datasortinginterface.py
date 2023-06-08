"""Primary class for converting experiment-specific behavior."""
from pynwb.file import NWBFile

from neuroconv.basedatainterface import BaseDataInterface

class SimulatedDataSortingInterface(BaseDataInterface):
    """Behavior interface for simulated_data conversion"""

    ExtractorName = "NumpySorting"
    
    def __init__(self):
        # This should load the data lazily and prepare variables you need
        

    def get_metadata(self):
        # Automatically retrieve as much metadata as possible
        metadata = super().get_metadata()
        
        return metadata