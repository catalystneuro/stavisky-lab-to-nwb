"""Primary NWBConverter class for this dataset."""
import inspect
from pathlib import Path
from pynwb import NWBFile
from typing import Optional, Union

from neuroconv import NWBConverter
from neuroconv.utils import load_dict_from_file, dict_deep_update
from neuroconv.tools.nwb_helpers import make_or_load_nwbfile

from stavisky_lab_to_nwb.general_interfaces import (
    StaviskyTemporalAlignmentInterface,
    StaviskyRecordingInterface,
    StaviskySortingInterface,
    StaviskySpikingBandPowerInterface,
    StaviskyFilteredRecordingInterface,
    StaviskySmoothedSpikingBandPowerInterface,
    StaviskySmoothedThreshCrossingInterface,
    StaviskyAudioInterface,
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
        Sorting=StaviskySortingInterface,
        Trials=BrainToTextTrialsInterface,
        Audio=StaviskyAudioInterface,
        PhonemeLogits=BrainToTextPhonemeLogitsInterface,
        DecodedText=BrainToTextDecodedTextInterface,
        SpikingBandPower10ms=StaviskySpikingBandPowerInterface,
        SpikingBandPower20ms=StaviskySpikingBandPowerInterface,
        FilteredRecording=StaviskyFilteredRecordingInterface,
        SmoothedSpikingBandPower=StaviskySmoothedSpikingBandPowerInterface,
        SmoothedThreshCrossing=StaviskySmoothedThreshCrossingInterface,
    )

    def __init__(
        self,
        conversion_config_path: Union[str, Path],
        port: int,
        host: str,
        source_data: dict,
        verbose: bool = True,
        session_start_time: float = 0.0,
        exclude_interfaces: list[str] = [],
        reuse_timestamps: bool = True,
    ):
        self.verbose = verbose
        default_config = load_dict_from_file(conversion_config_path)
        default_source_data = {
            interface_name: default_config[interface_name].get("source_data", {})
            for interface_name in default_config.keys()
            if interface_name not in exclude_interfaces
        }
        default_conversion_options = {
            interface_name: default_config[interface_name].get("conversion_options", {})
            for interface_name in default_config.keys()
            if interface_name not in exclude_interfaces
        }
        for name, data_interface in self.data_interface_classes.items():
            if name not in default_source_data:
                continue
            if "port" in inspect.signature(data_interface).parameters.keys():
                if "port" not in default_source_data[name]:
                    default_source_data[name]["port"] = port
                    default_source_data[name]["host"] = host
        source_data = dict_deep_update(default_source_data, source_data)
        self._validate_source_data(source_data=source_data, verbose=self.verbose)
        self.session_start_time = session_start_time
        self.data_interface_objects = {}
        timestamp_source_interfaces = {}
        for name, data_interface in self.data_interface_classes.items():
            if name not in source_data:
                continue
            interface_source_data = source_data[name]
            if reuse_timestamps and issubclass(data_interface, StaviskyTemporalAlignmentInterface):
                stream_name = interface_source_data.get("stream_name", None)
                if (stream_name is not None) and (stream_name in timestamp_source_interfaces):
                    print(
                        f"Skipping loading timestamps for {name}, taking timestamps from "
                        f"{timestamp_source_interfaces[stream_name]} instead"
                    )
                    source_data["load_timestamps"] = False
                    self.data_interface_objects[name] = data_interface(**interface_source_data)
                    self.data_interface_objects[name].set_timestamps_from_interface(
                        interface=self.data_interface_objects[timestamp_source_interfaces[stream_name]],
                    )
                else:
                    self.data_interface_objects[name] = data_interface(**interface_source_data)
                    timestamp_source_interfaces[stream_name] = name
            else:
                self.data_interface_objects[name] = data_interface(**interface_source_data)
        self.default_conversion_options = default_conversion_options

    def run_conversion(
        self,
        nwbfile_path: Optional[str] = None,
        nwbfile: Optional[NWBFile] = None,
        metadata: Optional[dict] = None,
        overwrite: bool = False,
        stub_test: bool = False,
        conversion_options: Optional[dict] = None,
    ) -> None:
        conversion_options = dict_deep_update(self.default_conversion_options, conversion_options)
        for name, data_interface in self.data_interface_objects.items():
            if "stub_test" in inspect.signature(data_interface.add_to_nwbfile).parameters.keys():
                if "stub_test" not in conversion_options[name]:
                    conversion_options[name]["stub_test"] = stub_test
        super().run_conversion(
            nwbfile_path=nwbfile_path,
            nwbfile=nwbfile,
            metadata=metadata,
            overwrite=overwrite,
            conversion_options=conversion_options,
        )

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
            if "Audio" in self.data_interface_objects:
                self.data_interface_objects["Trials"].align_by_interpolation(
                    unaligned_timestamps=nsp_analog_clock,
                    aligned_timestamps=redis_analog_clock,
                    clock="nsp_analog",
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
        # align other data fields
        if "PhonemeLogits" in self.data_interface_objects:
            self.data_interface_objects["PhonemeLogits"].set_aligned_starting_time(-self.session_start_time)
        if "DecodedText" in self.data_interface_objects:
            self.data_interface_objects["DecodedText"].set_aligned_starting_time(-self.session_start_time)
