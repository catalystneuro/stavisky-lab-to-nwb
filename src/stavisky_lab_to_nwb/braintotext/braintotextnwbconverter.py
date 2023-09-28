"""Primary NWBConverter class for this dataset."""
from neuroconv import NWBConverter
from pynwb import NWBFile
from typing import Optional

from neuroconv.tools.nwb_helpers import make_or_load_nwbfile

from stavisky_lab_to_nwb.general_interfaces import (
    StaviskyTemporalAlignmentInterface,
    StaviskyRecordingInterface,
    StaviskySpikingBandPowerInterface,
)

from stavisky_lab_to_nwb.braintotext import (
    BrainToTextPhonemeLogitsInterface,
    BrainToTextDecodedTextInterface,
    BrainToTextTrialsInterface,
)


class BrainToTextNWBConverter(NWBConverter):
    """Primary conversion class for my extracellular electrophysiology dataset."""

    data_interface_classes = dict(
        Recording=StaviskyRecordingInterface,
        # Sorting=StaviskySortingInterface,
        Trials=BrainToTextTrialsInterface,
        SpikingBandPower10ms=StaviskySpikingBandPowerInterface,
        SpikingBandPower20ms=StaviskySpikingBandPowerInterface,
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
        redis_neural_clock = None
        nsp_neural_clock = None
        # align recording start to session start time
        if "Recording" in self.data_interface_objects:
            redis_neural_clock = self.data_interface_objects["Recording"].get_timestamps(nsp=False)
            redis_neural_clock = (redis_neural_clock - self.session_start_time).astype("float64")
            nsp_neural_clock = self.data_interface_objects["Recording"].get_timestamps(nsp=True).astype("float64")
            self.data_interface_objects["Recording"].set_aligned_timestamps(redis_neural_clock, nsp=False)
        # align trial times
        if "Trials" in self.data_interface_objects:
            self.data_interface_objects["Trials"].set_aligned_starting_time(-self.session_start_time, clock="redis")
            if "Recording" in self.data_interface_objects:
                self.data_interface_objects["Trials"].align_by_interpolation(
                    unaligned_timestamps=nsp_neural_clock,
                    aligned_timestamps=redis_neural_clock,
                    clock="nsp_neural",
                )
        # align other ecephys data
        exclude_interfaces = ["Recording", "Audio", "Sorting", "Trials"]
        for name, interface in self.data_interface_objects.items():
            if not isinstance(interface, StaviskyTemporalAlignmentInterface):
                continue
            if name not in exclude_interfaces:
                interface.set_aligned_starting_time(-self.session_start_time, nsp=False)
                interface.set_dtype("float64", nsp=False)
                if "Recording" in self.data_interface_objects:
                    interface.align_by_interpolation(
                        unaligned_timestamps=nsp_neural_clock,
                        aligned_timestamps=redis_neural_clock,
                        nsp=True,
                    )
                else:
                    interface.set_aligned_timestamps(None, nsp=True)
