"""Primary NWBConverter class for this dataset."""
from neuroconv import NWBConverter
from pynwb import NWBFile
from typing import Optional

from neuroconv.tools.nwb_helpers import make_or_load_nwbfile

from stavisky_lab_to_nwb.general_interfaces import (
    StaviskyRecordingInterface,
    StaviskySortingInterface,
)

from stavisky_lab_to_nwb.braintotext import (
    BrainToTextPhonemeLogitsInterface,
    BrainToTextDecodedTextInterface,
    BrainToTextTrialsInterface,
    BrainToTextAudioInterface,
)


class BrainToTextNWBConverter(NWBConverter):
    """Primary conversion class for my extracellular electrophysiology dataset."""

    data_interface_classes = dict(
        Recording=StaviskyRecordingInterface,
        Sorting=StaviskySortingInterface,
        Trials=BrainToTextTrialsInterface,
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
        redis_neural_clock = None
        nsp_neural_clock = None
        # align audio start to session start time
        if "Audio" in self.data_interface_objects:
            redis_analog_clock = self.data_interface_objects["Audio"].get_timestamps(nsp=False)
            redis_analog_clock = (redis_analog_clock - self.session_start_time).astype("float64")
            nsp_analog_clock = self.data_interface_objects["Audio"].get_timestamps(nsp=True).astype("float64")
            self.data_interface_objects["Audio"].set_aligned_timestamps(redis_analog_clock, nsp=False)
        # align recording start to session start time
        if "Recording" in self.data_interface_objects:
            redis_neural_clock = self.data_interface_objects["Recording"].get_timestamps(nsp=False)
            redis_neural_clock = (redis_neural_clock - self.session_start_time).astype("float64")
            nsp_neural_clock = self.data_interface_objects["Recording"].get_timestamps(nsp=True).astype("float64")
            self.data_interface_objects["Recording"].set_aligned_timestamps(redis_neural_clock, nsp=False)
        # align sorting timestamps by recording or session start time
        if "Sorting" in self.data_interface_objects:
            if "Recording" in self.data_interface_objects:
                self.data_interface_objects["Sorting"].align_by_interpolation(
                    unaligned_timestamps=nsp_neural_clock,
                    aligned_timestamps=redis_neural_clock,
                    nsp=True,
                )
            else:
                self.data_interface_objects["Sorting"].sorting_extractor.set_clock("redis")
                self.data_interface_objects["Sorting"].set_aligned_starting_time(-self.session_start_time, nsp=False)
                self.data_interface_objects["Sorting"].set_dtype("float64", nsp=False)
        # align trial times
        if "Trials" in self.data_interface_objects:
            self.data_interface_objects["Trials"].set_aligned_starting_time(-self.session_start_time, clock="redis")
            if "Recording" in self.data_interface_objects:
                self.data_interface_objects["Trials"].align_by_interpolation(
                    unaligned_timestamps=nsp_neural_clock,
                    aligned_timestamps=redis_neural_clock,
                    clock="nsp_neural",
                )
        # align other data fields
        if "PhonemeLogits" in self.data_interface_objects:
            self.data_interface_objects["PhonemeLogits"].set_aligned_starting_time(-self.session_start_time)
        if "DecodedText" in self.data_interface_objects:
            self.data_interface_objects["DecodedText"].set_aligned_starting_time(-self.session_start_time)
