"""Primary NWBConverter class for this dataset."""
from neuroconv import NWBConverter
from pynwb import NWBFile
from typing import Optional

from neuroconv.tools.nwb_helpers import make_or_load_nwbfile

from stavisky_lab_to_nwb.braintotext import (
    BrainToTextPhonemeLogitsInterface,
    BrainToTextDecodedTextInterface,
    BrainToTextTrialsInterface,
    BrainToTextAudioInterface,
)


class BrainToTextNWBConverter(NWBConverter):
    """Primary conversion class for my extracellular electrophysiology dataset."""

    data_interface_classes = dict(
        # Recording=StaviskyRecordingInterface,
        # Sorting=StaviskySortingInterface,
        # Trials=StaviskyTrialsInterface,
        # SpikingBandPower1ms=StaviskySpikingBandPowerInterface,
        # SpikingBandPower20ms=StaviskySpikingBandPowerInterface,
        Audio=BrainToTextAudioInterface,
        PhonemeLogits=BrainToTextPhonemeLogitsInterface,
        DecodedText=BrainToTextDecodedTextInterface,
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
        # initialize common clock variables
        redis_analog_clock = None
        nsp_analog_clock = None
        # align audio start to session start time
        if "Audio" in self.data_interface_objects:
            redis_analog_clock = self.data_interface_objects["Audio"].get_timestamps(nsp=False)
            redis_analog_clock = (redis_analog_clock - self.session_start_time).astype("float64")
            nsp_analog_clock = self.data_interface_objects["Audio"].get_timestamps(nsp=True).astype("float64")
            self.data_interface_objects["Audio"].set_aligned_timestamps(redis_analog_clock, nsp=False)
