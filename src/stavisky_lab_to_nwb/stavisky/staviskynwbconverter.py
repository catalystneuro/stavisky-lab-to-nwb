"""Primary NWBConverter class for this dataset."""
from neuroconv import NWBConverter
from pynwb import NWBFile
from typing import Optional

from neuroconv.tools.nwb_helpers import make_or_load_nwbfile

from stavisky_lab_to_nwb.stavisky import (
    StaviskyPhonemeLogitsInterface,
    StaviskyDecodedTextInterface,
    # StaviskySpikingBandPowerInterface,
    # RedisSortingInterface,
    StaviskyRecordingInterface,
    StaviskyTrialsInterface,
)


class StaviskyNWBConverter(NWBConverter):
    """Primary conversion class for my extracellular electrophysiology dataset."""

    data_interface_classes = dict(
        Recording=StaviskyRecordingInterface,
        # Sorting=RedisSortingInterface,
        Trials=StaviskyTrialsInterface,
        PhonemeLogits=StaviskyPhonemeLogitsInterface,
        DecodedText=StaviskyDecodedTextInterface,
    )

    def __init__(
        self,
        source_data: dict,
        verbose: bool = True,
        session_start_time: float = 0.0,
    ):
        super().__init__(source_data=source_data, verbose=verbose)
        self.session_start_time = session_start_time

    def temporally_align_data_interfaces(self):
        if self.session_start_time != 0.0:
            self.data_interface_objects["Recording"].set_aligned_starting_time(-self.session_start_time)
