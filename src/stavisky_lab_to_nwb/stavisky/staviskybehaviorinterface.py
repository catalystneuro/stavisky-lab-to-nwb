"""Primary class for converting experiment-specific behavior."""
from pynwb import NWBFile
from ndx_events import LabeledEvents

from neuroconv.basedatainterface import BaseDataInterface

class StaviskyPhonemeLogitsInterface(BaseDataInterface):
    """Decoding phoneme logits interface for Stavisky Redis conversion""""
    
    def __init__(
        self, 
        port: int,
        host: str,
        timestamps: Optional[np.ndarray] = None,
        smooth_timestamps: bool = False,
    ):
        # This should load the data lazily and prepare variables you need
        super().__init__(port=port, host=host)
        self._timestamps = timestamps
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
        
        start_time = metadata["NWBFile"]["session_start_time"].timestamp()
        
        # prepare timestamps if provided
        build_timestamps = self._timestamps is None
        if build_timestamps:
            timestamps = []
        else:
            timestamps = self._timestamps
            
        # TODO: read in chunks?
        decoder_output = r.xrange("binned:decoderOutput:stream")
        logits = []
        for entry in decoder_output:
            if b'start' in entry[1].keys():
                trial_logits = []
            elif b'end' in entry[1].keys():
                if build_timestamps:
                    if self._smooth_timestamps:
                        start_time = int(trial_logits[0][0].split(b'-')[0]) / 1000. - start_time
                        stop_time = int(trial_logits[-1][0].split(b'-')[0]) / 1000  - start_time
                        trial_timestamps = np.linspace(start_time, stop_time, len(trial_logits))
                    else:
                        trial_timestamps = np.array([(int(tl[0].split(b'-')[0]) / 1000.  - start_time) for tl in trial_logits])
                    timestamps.append(trial_timestamps)
                trial_logits = np.stack([np.frombuffer(tl[b'logits'], dtype=np.float32) for tl in trial_logits[1:-1]], axis=0)
                logits.append(trial_logits)
                trial_logits = []
            elif b'logits' in entry[1].keys():
                trial_logits.append(entry)
        logits = np.concatenate(logits, axis=0)
        if build_timestamps:
            timestamps = np.concatenate(timestamps)
        
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


class StaviskyDecodedTextInterface(BaseDataInterface):
    """Decoded text interface for Stavisky Redis conversion""""

    def __init__(
        self, 
        port: int,
        host: str,
    ):
        # This should load the data lazily and prepare variables you need
        super().__init__(port=port, host=host)

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
        
        start_time = metadata["NWBFile"]["session_start_time"].timestamp()
        
        decoder_output = r.xrange("binned:decoderOutput:stream")
        decoded_timestamps = []
        decoded_text = []
        last_text = ""
        for entry in decoder_output:
            if b'partial_decoded_sentence' in entry[1].keys():
                curr_text = str(entry[1][b'partial_decoded_sentence'], "utf-8").strip()
                assert len(curr_text) >= len(last_text)
                if len(curr_text) > len(last_text):
                    new_word = curr_text.split()[-1].lower()
                    timestamp = int(entry[0].split(b'-')[0]) / 1000. - start_time
                    decoded_text.append(new_word)
                    decoded_timestamps.append(timestamp)
                    last_text = curr_text
            elif b'final_decoded_sentence' in entry[1].keys():
                last_text = ""
                timestamp = int(entry[0].split(b'-')[0]) / 1000. - start_time
                # TODO: determine if this is necessary. does the model decide when to stop,
                # or is it externally controlled?
                decoded_text.append("STOP")
                decoded_timestamps.append(timestamp)
        
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
