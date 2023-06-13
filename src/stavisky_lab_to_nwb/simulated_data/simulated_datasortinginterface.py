"""Primary class for converting sorting data."""
import redis
import numpy as np
from pynwb.file import NWBFile
from typing import Union, Optional, List, Tuple, Sequence, Literal

from neuroconv.datainterfaces.ecephys.basesortingextractorinterface import BaseSortingExtractorInterface


class RedisSortingInterface(BaseSortingExtractorInterface):
    """Sorting interface for Stavisky Redis conversion"""
    
    def __init__(self):
        # This should load the data lazily and prepare variables you need

    def get_metadata(self):
        # Automatically retrieve as much metadata as possible
        metadata = super().get_metadata()
        
        return metadata

    def run_conversion(self, nwbfile: NWBFile, metadata: dict):
        # All the custom code to write to PyNWB

        return nwbfile
