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
    ):
        super().__init__(source_data=source_data, verbose=verbose)
    
    def run_conversion(
        self,
        nwbfile_path: Optional[str] = None,
        nwbfile: Optional[NWBFile] = None,
        metadata: Optional[dict] = None,
        overwrite: bool = False,
        conversion_options: Optional[dict] = None,
    ) -> None:
        # TODO: ideally we'd have an `alignment_options` arg or something like that
        # instead of completely reimplementing `run_conversion`
        if metadata is None:
            metadata = self.get_metadata()

        self.validate_metadata(metadata=metadata)

        self.validate_conversion_options(conversion_options=conversion_options)

        self.temporally_align_data_interfaces(metadata=metadata)

        with make_or_load_nwbfile(
            nwbfile_path=nwbfile_path,
            nwbfile=nwbfile,
            metadata=metadata,
            overwrite=overwrite,
            verbose=self.verbose,
        ) as nwbfile_out:
            self.add_to_nwbfile(nwbfile_out, metadata, conversion_options)

    def temporally_align_data_interfaces(self, metadata=None):
        if metadata is not None:
            session_start_time = metadata["NWBFile"]["session_start_time"].timestamp()
            self.data_interface_objects["Recording"].set_aligned_starting_time(-session_start_time)