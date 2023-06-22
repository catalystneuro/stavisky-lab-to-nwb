"""Primary NWBConverter class for this dataset."""
from neuroconv import NWBConverter

from stavisky_lab_to_nwb.stavisky import (
    # StaviskyPhonemeLogitsInterface,
    # StaviskyDecodedTextInterface,
    # StaviskySpikingBandPowerInterface,
    # RedisSortingInterface,
    # RedisRecordingInterface,
    StaviskyTrialsInterface,
)


class StaviskyNWBConverter(NWBConverter):
    """Primary conversion class for my extracellular electrophysiology dataset."""

    data_interface_classes = dict(
        # Recording=RedisRecordingInterface,
        # Sorting=RedisSortingInterface,
        Trials=StaviskyTrialsInterface,
    )

    def __init__(
        self,
        source_data: dict,
        verbose: bool = True,
    ):
        super().__init__(source_data=source_data, verbose=verbose)
