"""Primary class for converting experiment-specific behavior."""
import redis
import numpy as np
from pynwb import NWBFile, TimeSeries
from ndx_events import LabeledEvents
from typing import Optional

from neuroconv.basedatainterface import BaseDataInterface
from neuroconv.tools.nwb_helpers import get_module

class StaviskyPhonemeLogitsInterface(BaseDataInterface):
    """Decoding phoneme logits interface for Stavisky Redis conversion"""
    
    def __init__(
        self, 
        port: int,
        host: str,
        timestamps: Optional[list] = None,
        smooth_timestamps: bool = False,
    ):
        # This should load the data lazily and prepare variables you need
        super().__init__(port=port, host=host)
        self._timestamps = None if timestamps is None else np.array(timestamps)
        self._smooth_timestamps = smooth_timestamps

    def get_metadata(self):
        # Automatically retrieve as much metadata as possible
        metadata = super().get_metadata()
        
        return metadata

    def add_to_nwbfile(self, nwbfile: NWBFile, metadata: dict):
        # All the custom code to write to PyNWB
        r = redis.Redis(
            port=self.source_data["port"],
            host=self.source_data["host"],
        )
        r.ping()
        
        module_name = "Speech decoding"
        module_description = "Contains decoder outputs for real-time speech decoding"
        processing_module = get_module(nwbfile=nwbfile, name=module_name, description=module_description)
        
        # session_start_time = metadata["NWBFile"]["session_start_time"].timestamp()
        session_start_time = np.frombuffer(r.xrange("metadata")[0][1][b"startTime"], dtype=np.float64).item()
        
        # prepare timestamps if provided
        build_timestamps = self._timestamps is None
        timestamps = []
        if not build_timestamps:
            reference_timestamps = [
                int(entry[0].split(b'-')[0]) / 1000. for entry in r.xrange("binnedFeatures_20ms")]
            reference_timestamps = np.array(reference_timestamps[3::4])
            
        # TODO: read in chunks?
        decoder_output = r.xrange("binned:decoderOutput:stream")
        logits = []
        for entry in decoder_output:
            if b'start' in entry[1].keys():
                trial_logits = []
            elif b'end' in entry[1].keys():
                if self._smooth_timestamps and build_timestamps:
                    start_time = int(trial_logits[0][0].split(b'-')[0]) / 1000. - session_start_time
                    stop_time = int(trial_logits[-1][0].split(b'-')[0]) / 1000  - session_start_time
                    trial_timestamps = np.linspace(start_time, stop_time, len(trial_logits - 2))
                else:
                    trial_timestamps = np.array([(int(tl[0].split(b'-')[0]) / 1000. - session_start_time) for tl in trial_logits])
                if build_timestamps:
                    timestamps.append(trial_timestamps)
                else:
                    start_idx = np.nonzero(np.diff(trial_timestamps[0] > reference_timestamps))[0][0]
                    search_end = start_idx + 10
                    while start_idx < search_end:
                        diff = trial_timestamps - reference_timestamps[start_idx:(start_idx+len(trial_timestamps))]
                        if np.all(diff > 0):
                            break
                        start_idx += 1
                    if start_idx == search_end:
                        raise AssertionError("Can't find start idx")
                    timestamps.append(
                        self._timestamps[start_idx:(start_idx+len(trial_timestamps))])
                
                trial_logits = np.stack([np.frombuffer(tl[1][b'logits'], dtype=np.float32) for tl in trial_logits], axis=0)
                logits.append(trial_logits)
                trial_logits = []
            elif b'logits' in entry[1].keys():
                trial_logits.append(entry)
        logits = np.concatenate(logits, axis=0)
        timestamps = np.concatenate(timestamps)
        
        logits_timeseries = TimeSeries(
            name="phoneme_logits",
            data=logits,
            unit="n.a.",
            timestamps=timestamps,
            description="Log-probabilities of the 39 phonemes plus silence and space between words, as " + \
                "predicted by an RNN decoder",
        )
        
        processing_module.add_data_interface(logits_timeseries)
        
        r.close()
        
        return nwbfile


class StaviskyDecodedTextInterface(BaseDataInterface):
    """Decoded text interface for Stavisky Redis conversion"""

    def __init__(
        self, 
        port: int,
        host: str,
        timestamps: Optional[list] = None,
    ):
        # This should load the data lazily and prepare variables you need
        super().__init__(port=port, host=host)
        self._timestamps = None if timestamps is None else np.array(timestamps)

    def get_metadata(self):
        # Automatically retrieve as much metadata as possible
        metadata = super().get_metadata()
        
        return metadata

    def add_to_nwbfile(self, nwbfile: NWBFile, metadata: dict):
        # All the custom code to write to PyNWB
        r = redis.Redis(
            port=self.source_data["port"],
            host=self.source_data["host"],
        )
        r.ping()
        
        module_name = "Speech decoding"
        module_description = "Contains decoder outputs for real-time speech decoding"
        processing_module = get_module(nwbfile=nwbfile, name=module_name, description=module_description)
        
        # session_start_time = metadata["NWBFile"]["session_start_time"].timestamp()
        session_start_time = np.frombuffer(r.xrange("metadata")[0][1][b"startTime"], dtype=np.float64).item()
        
        if not self._timestamps is None:
            reference_timestamps = [
                int(entry[0].split(b'-')[0]) / 1000. for entry in r.xrange("binnedFeatures_20ms")]
            reference_timestamps = np.array(reference_timestamps[3::4])
        
        decoder_output = r.xrange("binned:decoderOutput:stream")
        decoded_timestamps = []
        decoded_text = []
        last_text = ""
        for entry in decoder_output:
            if b'partial_decoded_sentence' in entry[1].keys():
                curr_text = str(entry[1][b'partial_decoded_sentence'], "utf-8").strip()
                if curr_text != last_text:
                    timestamp = int(entry[0].split(b'-')[0]) / 1000. - session_start_time
                    if not self._timestamps is None:
                        timestamp_idx = np.nonzero(np.diff(timestamp > reference_timestamps))[0][0]
                        timestamp = self._timestamps[timestamp_idx]
                    decoded_text.append(curr_text)
                    decoded_timestamps.append(timestamp)
                    last_text = curr_text
            elif b'final_decoded_sentence' in entry[1].keys():
                last_text = ""
            #     timestamp = int(entry[0].split(b'-')[0]) / 1000. - session_start_time
            #     # TODO: determine if this is necessary. does the model decide when to stop,
            #     # or is it externally controlled? if the former, then a STOP event might be useful
            #     decoded_text.append("STOP")
            #     decoded_timestamps.append(timestamp)
        
        # get unique words
        dictionary = sorted(list(set(decoded_text)))
        decoded_text_idx = [dictionary.index(word) for word in decoded_text]
        
        events = LabeledEvents(
            name='decoded_text',
            description='Text decoded from neural activity using language model',
            timestamps=decoded_timestamps,
            resolution=1e-3,  # resolution of the timestamps, i.e., smallest possible difference between timestamps
            data=decoded_text_idx,
            labels=dictionary
        )
        
        processing_module.add_data_interface(events)
        
        return nwbfile

if __name__ == "__main__":
    interface = StaviskyDecodedTextInterface(
        port=6379,
        host="localhost",
    )
    interface.add_to_nwbfile(nwbfile=None, metadata={})