"""Primary class for converting experiment-specific behavior."""
import redis
import numpy as np
from pynwb.file import NWBFile, TimeSeries

from neuroconv.basedatainterface import BaseDataInterface
from neuroconv.tools.nwb_helpers import get_module

class SimulatedDataPhonemeLogitsInterface(BaseDataInterface):
    """Behavior interface for simulated_data conversion"""

    def __init__(
        self, 
        port: int,
        host: str
    ):
        # This should load the data lazily and prepare variables you need
        super().__init__(port=port, host=host)

    def get_metadata(self):
        # Automatically retrieve as much metadata as possible
        metadata = super().get_metadata()
        
        return metadata

    def run_conversion(self, nwbfile: NWBFile, metadata: dict):
        # All the custom code to write to PyNWB
        r = redis.Redis(
            port=self.source_data["port"],
            host=self.source_data["host"],
        )
        r.ping()
        
        module_name = "Speech decoding"
        module_description = "Contains decoder outputs for real-time speech decoding"
        processing_module = get_module(nwbfile=nwbfile, name=module_name, description=module_description)
        
        decoder_output = r.xrange("binned:decoderOutput:stream")
        timestamps = []
        logits = []
        for entry in decoder_output:
            if b'start' in entry[1].keys():
                trial_logits = []
            if b'end' in entry[1].keys():
                start_time = int(trial_logits[0][0].split(b'-')[0]) / 1000.
                stop_time = int(trial_logits[-1][0].split(b'-')[0]) / 1000.
                trial_timestamps = np.linspace(start_time, stop_time, len(trial_logits))
                trial_logits = np.stack([np.frombuffer(tl[b'logits'], dtype=np.float32) for tl in trial_logits[1:-1]], axis=0)
                timestamps.append(trial_timestamps)
                logits.append(trial_logits)
                trial_logits = []
            else:
                trial_logits.append(entry)
        timestamps = np.concatenate(timestamps) - metadata["NWBFile"]["session_start_time"].timestamp()
        logits = np.concatenate(logits, axis=0)
        
        logits_timeseries = TimeSeries(
            name="phoneme_logits",
            data=logits,
            unit="",
            timestamps=timestamps,
            description="Log-probabilities of the 39 phonemes plus silence and space between words, as " + \
                "predicted by an RNN decoder",
        )
        
        processing_module.add_data_interface(logits_timeseries)
        
        r.close()
        
        return nwbfile
