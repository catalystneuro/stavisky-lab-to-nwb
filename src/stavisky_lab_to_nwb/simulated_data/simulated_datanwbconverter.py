"""Primary NWBConverter class for this dataset."""
from neuroconv import NWBConverter
from neuroconv.datainterfaces import (
    SpikeGLXRecordingInterface,
    SpikeGLXLFPInterface,
    PhySortingInterface,
)

from stavisky_lab_to_nwb.simulated_data import (
    # SimulatedDataBehaviorInterface,
    # SimulatedDataSortingInterface,
    SimulatedDataRecordingInterface,
    SimulatedDataTrialsInterface,
)

class SimulatedDataNWBConverter(NWBConverter):
    """Primary conversion class for my extracellular electrophysiology dataset."""

    data_interface_classes = dict(
        Recording=SimulatedDataRecordingInterface,
        # Sorting=SimulatedDataSortingInterface,
        # Behavior=SimulatedDataBehaviorInterface,
        Trials=SimulatedDataTrialsInterface,
    )
    
    def __init__(
        self,
        source_data: dict,
        verbose: bool = True,
    ):
        super().__init__(source_data=source_data, verbose=verbose)

