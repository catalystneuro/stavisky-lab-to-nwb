"""Primary NWBConverter class for this dataset."""
from neuroconv import NWBConverter

from stavisky_lab_to_nwb.simulated_data import (
    # StaviskyPhonemeLogitsInterface, 
    # StaviskyDecodedTextInterface,
    StaviskySpikingBandPowerInterface,
    StaviskySortingInterface,
    StaviskyRecordingInterface,
    StaviskyTrialsInterface,
)

class StaviskyNWBConverter(NWBConverter):
    """Primary conversion class for my extracellular electrophysiology dataset."""

    data_interface_classes = dict(
        Recording=StaviskyRecordingInterface,
        Sorting=StaviskySortingInterface,
        SpikingBandPower=StaviskySpikingBandPowerInterface,
        # Trials=StaviskyTrialsInterface,
    )
    
    def __init__(
        self,
        source_data: dict,
        verbose: bool = True,
    ):
        super().__init__(source_data=source_data, verbose=verbose)
        
        recording = self.data_interface_objects["Recording"]
        self.data_interface_objects["Sorting"].register_recording(recording, frequency_ratio=30, override_frequency=True)
        self.data_interface_objects["SpikingBandPower"].register_recording(recording, frequency_ratio=30)

