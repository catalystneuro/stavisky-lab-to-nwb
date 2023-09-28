"""Primary class for converting trial data."""
import redis
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Literal

from pynwb import NWBFile

from neuroconv.datainterfaces.text.timeintervalsinterface import TimeIntervalsInterface
from neuroconv.basedatainterface import BaseDataInterface


class BrainToTextTrialsInterface(TimeIntervalsInterface):
    def __init__(
        self,
        port: int,
        host: str,
        read_kwargs: Optional[dict] = None,
        verbose: bool = True,
    ):
        """
        Parameters
        ----------
        file_path : FilePath
        read_kwargs : dict, optional
        verbose : bool, default: True
        """
        read_kwargs = read_kwargs or dict()
        super(TimeIntervalsInterface, self).__init__(port=port, host=host)
        self.verbose = verbose

        self._read_kwargs = read_kwargs
        self.dataframe = self._read_file(port=port, host=host, **read_kwargs)
        self.time_intervals = None

    def get_metadata(self) -> dict:
        metadata = super(TimeIntervalsInterface, self).get_metadata()
        metadata["TimeIntervals"] = dict(
            trials=dict(
                table_name="trials",
                table_description=f"experimental trials generated from redis trial info stream",
            )
        )

        return metadata

    def _read_file(
        self,
        port: int,
        host: str,
    ):
        # Instantiate Redis client and check connection
        r = redis.Redis(port=port, host=host)
        r.ping()

        # Extract trial information
        trial_info = r.xrange("trial_info")
        final_decoded_sentence = r.xrange("tts_final_decoded_sentence")
        tts_info = r.xrange("tts_node_info")
        assert len(trial_info) == len(final_decoded_sentence)
        assert len(trial_info) == len(tts_info)

        # Parse and store "trial_info" by trial ID
        # TODO: make conversion factors to seconds for different clocks as arguments??
        trial_list = []
        trial_ids = []
        for n, (timestamp, trial) in enumerate(trial_info):
            trial_ids.append(int(trial[b"trial_num"]))
            trial_list.append(
                {
                    "start_time": np.frombuffer(trial[b"trial_start_redis_time"], dtype=np.int64).item() / 1.0e3,
                    "stop_time": np.frombuffer(trial[b"trial_end_redis_time"], dtype=np.int64).item() / 1.0e3,
                    "go_cue_time": np.frombuffer(trial[b"go_cue_redis_time"], dtype=np.int64).item() / 1.0e3,
                    "delay_duration": float(trial[b"delay_duration"]),
                    "inter_trial_duration": float(trial[b"inter_trial_duration"]),
                    "sentence_cue": trial[b"cue"].decode("utf-8"),
                    "ended_with_pause": int(trial[b"ended_with_pause"]),
                    "ended_with_timeout": int(trial[b"ended_with_timeout"]),
                    "start_nsp_neural_time": np.frombuffer(trial[b"trial_start_nsp_neural_time"], dtype=np.int64).item()
                    / 3.0e4,
                    "start_nsp_analog_time": np.frombuffer(trial[b"trial_start_nsp_analog_time"], dtype=np.int64).item()
                    / 1.0e9,
                    "go_cue_nsp_neural_time": np.frombuffer(trial[b"go_cue_nsp_neural_time"], dtype=np.int64).item()
                    / 3.0e4,
                    "go_cue_nsp_analog_time": np.frombuffer(trial[b"go_cue_nsp_analog_time"], dtype=np.int64).item()
                    / 1.0e9,
                    "stop_nsp_neural_time": np.frombuffer(trial[b"trial_end_nsp_neural_time"], dtype=np.int64).item()
                    / 3.0e4,
                    "stop_nsp_analog_time": np.frombuffer(trial[b"trial_end_nsp_analog_time"], dtype=np.int64).item()
                    / 1.0e9,
                    "decoded_sentence": final_decoded_sentence[n][1][b"final_decoded_sentence"].decode("utf-8"),
                    "tts_stop_time": int(tts_info[n][0].decode("utf-8").split("-")[0]) / 1.0e3,
                }
            )

        return pd.DataFrame([trial_list[i] for i in np.argsort(trial_ids)])

    def add_to_nwbfile(
        self,
        nwbfile: NWBFile,
        metadata: Optional[dict] = None,
        tag: str = "trials",
    ):
        column_descriptions = {
            "start_nsp_neural_time": "Time of trial start on nsp neural clock",
            "start_nsp_analog_time": "Time of trial start on nsp analog clock",
            "go_cue_time": "Time of presentation of go cue",
            "go_cue_nsp_neural_time": "Time of presentation of go cue",
            "go_cue_nsp_analog_time": "Time of presentation of go cue",
            "stop_nsp_neural_time": "Time of trial end on nsp neural clock",
            "stop_nsp_analog_time": "Time of trial end on nsp analog clock",
            "sentence_cue": "The sentence cue that the subject speaks/imagines speaking",
            "delay_duration": "Duration of delay period before go cue",
            "inter_trial_duration": "Duration of interval between end of current trial and start of next trial",
            "ended_with_pause": "Whether the trial ended with pause",
            "ended_with_timeout": "Whether the trial ended with timeout",
            "decoded_sentence": "The sentence decoded by the brain-to-text decoder",
            "tts_stop_time": "Stop time for completion of text-to-speech playback of decoded sentence",
        }

        return super().add_to_nwbfile(
            nwbfile=nwbfile,
            metadata=metadata,
            tag=tag,
            column_descriptions=column_descriptions,
        )

    def set_aligned_starting_time(
        self,
        aligned_starting_time: float,
        clock: Optional[Literal["redis", "nsp_neural", "nsp_analog"]] = None,
    ):
        if clock is not None:
            assert clock in ["redis", "nsp_neural", "nsp_analog"]
            if clock == "redis":
                clock = None
        include_clock = (clock or "") + "_time"
        exclude_clock = set(["nsp_neural", "nsp_analog"]) - set([clock])
        timing_columns = [
            column
            for column in self.dataframe.columns
            if (column.endswith(include_clock) and not any([clock_name in column for clock_name in exclude_clock]))
        ]

        for column in timing_columns:
            self.dataframe[column] += aligned_starting_time

    def align_by_interpolation(
        self,
        unaligned_timestamps: np.ndarray,
        aligned_timestamps: np.ndarray,
        clock: Optional[Literal["redis", "nsp_neural", "nsp_analog"]] = None,
    ):
        if clock is not None:
            assert clock in ["redis", "nsp_neural", "nsp_analog"]
            if clock == "redis":
                clock = None
        include_clock = (clock or "") + "_time"
        exclude_clock = set(["nsp_neural", "nsp_analog"]) - set([clock])
        timing_columns = [
            column
            for column in self.dataframe.columns
            if (column.endswith(include_clock) and not any([clock_name in column for clock_name in exclude_clock]))
        ]

        for column in timing_columns:
            super().align_by_interpolation(
                unaligned_timestamps=unaligned_timestamps,
                aligned_timestamps=aligned_timestamps,
                column=column,
            )
