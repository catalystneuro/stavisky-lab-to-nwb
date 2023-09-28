"""Primary classes for converting Stavisky decoding data."""
import redis
import numpy as np
from pynwb import NWBFile, TimeSeries
from ndx_events import LabeledEvents
from typing import Optional
from hdmf.backends.hdf5 import H5DataIO

from neuroconv.basedatainterface import BaseDataInterface
from neuroconv.tools.nwb_helpers import get_module

from stavisky_lab_to_nwb.utils.redis import read_entry


class BrainToTextPhonemeLogitsInterface(BaseDataInterface):
    """Decoding phoneme logits interface for Stavisky Redis conversion"""

    def __init__(
        self,
        port: int,
        host: str,
        stream_name: str = "binned:decoderOutput:stream",
        data_field: str = "logits",
        dtype: str = "float32",
    ):
        super().__init__(
            port=port,
            host=host,
            stream_name=stream_name,
            data_field=data_field,
        )
        self.aligned_starting_time = 0.0
        self.dtype = dtype

    def set_aligned_starting_time(self, aligned_starting_time: float):
        """To imitate TemporalAlignmentInterface functionality without reading timestamps"""
        self.aligned_starting_time = aligned_starting_time

    def add_to_nwbfile(
        self,
        nwbfile: NWBFile,
        metadata: dict,
        chunk_size: Optional[int] = None,
    ):
        # initialize redis client and check connection
        r = redis.Redis(
            port=self.source_data["port"],
            host=self.source_data["host"],
        )
        r.ping()

        # get processing module
        module_name = "behavior"
        module_description = "Contains decoder outputs for real-time speech decoding."
        processing_module = get_module(nwbfile=nwbfile, name=module_name, description=module_description)

        # loop through data and make arrays
        decoder_output = r.xrange(self.source_data["stream_name"], count=chunk_size)
        logits = []
        timestamps = []
        while len(decoder_output) > 0:  # read in chunks until entire stream has been read
            for entry in decoder_output:
                # append data until end of each trial
                if b"start" in entry[1].keys():  # start of trial decoding period
                    trial_logits = []
                elif b"end" in entry[1].keys():  # end of trial decoding period
                    # timestamps
                    trial_timestamps = np.array(
                        [(int(tl[0].split(b"-")[0]) / 1000.0 + self.aligned_starting_time) for tl in trial_logits]
                    )
                    timestamps.append(trial_timestamps)
                    # decoding logits
                    trial_logits = np.stack(
                        [
                            read_entry(tl[1], self.source_data["data_field"], encoding="buffer", dtype=self.dtype)
                            for tl in trial_logits
                        ],
                        axis=0,
                    )
                    logits.append(trial_logits)
                    trial_logits = []  # reset data stack
                elif b"logits" in entry[1].keys():
                    trial_logits.append(entry)
            # read next `chunk_size` entries
            decoder_output = r.xrange(
                self.source_data["stream_name"], count=chunk_size, min=b"(" + decoder_output[-1][0]
            )
        # stack all data
        logits = np.concatenate(logits, axis=0)
        timestamps = np.concatenate(timestamps)

        # create timeseries obj
        logits_timeseries = TimeSeries(
            name="phoneme_logits",
            data=H5DataIO(logits, compression="gzip"),
            unit="n.a.",
            timestamps=H5DataIO(timestamps, compression="gzip"),
            description=(
                "Log-probabilities of the 39 phonemes plus silence and space between words, as "
                "predicted by an RNN decoder"
            ),
            comments=(
                "The 41 columns correspond to, in order, silence, the space between words, and "
                "the 39 phonemes of the CMU pronouncing dictionary in alphabetical order"
            ),
        )

        # add to processing module
        processing_module.add_data_interface(logits_timeseries)

        # close redis client
        r.close()

        return nwbfile


class BrainToTextDecodedTextInterface(BaseDataInterface):
    """Decoded text interface for Stavisky Redis conversion"""

    def __init__(
        self,
        port: int,
        host: str,
        stream_name: str = "binned:decoderOutput:stream",
    ):
        super().__init__(port=port, host=host, stream_name=stream_name)
        self.aligned_starting_time = 0.0

    def set_aligned_starting_time(self, aligned_starting_time: float):
        """To imitate TemporalAlignmentInterface functionality without reading timestamps"""
        self.aligned_starting_time = aligned_starting_time

    def add_to_nwbfile(
        self,
        nwbfile: NWBFile,
        metadata: dict,
        chunk_size: Optional[int] = None,
    ):
        # initialize redis client and check connection
        r = redis.Redis(
            port=self.source_data["port"],
            host=self.source_data["host"],
        )
        r.ping()

        # get processing module
        module_name = "behavior"
        module_description = "Contains decoder outputs for real-time speech decoding."
        processing_module = get_module(nwbfile=nwbfile, name=module_name, description=module_description)

        # loop through data and log decoder output
        decoder_output = r.xrange(self.source_data["stream_name"], count=chunk_size)
        decoded_timestamps = []
        decoded_text = []
        last_text = ""
        while len(decoder_output) > 0:  # read in chunks until entire stream has been read
            for entry in decoder_output:
                if b"partial_decoded_sentence" in entry[1].keys():
                    curr_text = str(entry[1][b"partial_decoded_sentence"], "utf-8").strip()
                    if curr_text != last_text:  # if decoded text changes, log it
                        timestamp = int(entry[0].split(b"-")[0]) / 1000.0 + self.aligned_starting_time
                        decoded_text.append(curr_text)
                        decoded_timestamps.append(timestamp)
                        last_text = curr_text
                elif b"final_decoded_sentence" in entry[1].keys():  # reset on trial end
                    last_text = ""
            # read next `chunk_size` entries
            decoder_output = r.xrange(
                self.source_data["stream_name"], count=chunk_size, min=b"(" + decoder_output[-1][0]
            )
        # stack timestamps
        decoded_timestamps = np.array(decoded_timestamps, dtype="float64")

        # get unique decoder outputs
        output_set = sorted(list(set(decoded_text)))
        decoded_text_idx = [output_set.index(sentence) for sentence in decoded_text]

        # cast to lower int dtype if possible
        if np.max(decoded_text_idx) < 65536:
            decoded_text_idx = np.array(decoded_text_idx, dtype="uint16")
        else:
            # can't imagine it'll exceed 4294967295
            decoded_text_idx = np.array(decoded_text_idx, dtype="uint32")

        # make labeledevents obj
        events = LabeledEvents(
            name="decoded_text",
            description="Text decoded from RNN-predicted phonemes using language model.",
            timestamps=H5DataIO(decoded_timestamps, compression="gzip"),
            data=H5DataIO(decoded_text_idx, compression="gzip"),
            labels=output_set,
        )

        # add to processing module
        processing_module.add_data_interface(events)

        # close redis client
        r.close()

        return nwbfile
