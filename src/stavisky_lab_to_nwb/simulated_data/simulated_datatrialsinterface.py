import redis
import numpy as np
from pathlib import Path
from typing import Optional

from pynwb.file import NWBFile

from neuroconv.basedatainterface import BaseDataInterface


class SimulatedDataTrialsInterface(BaseDataInterface):
    """Trials interface for simulated_data conversion"""
    
    def __init__(
        self, 
        port: int,
        host: str,
    ):
        super().__init__(port=port, host=host)

    def get_metadata(self):
        # Automatically retrieve as much metadata as possible
        metadata = super().get_metadata()
        
        return metadata
    
    def run_conversion(
        self,
        nwbfile: NWBFile,
        metadata: Optional[dict] = None,
        stub_test: bool = False,
    ):
        # All the custom code to write to PyNWB
        r = redis.Redis(
            port=self.source_data["port"],
            host=self.source_data["host"],
        )
        r.ping()
        
        session_start_time = np.frombuffer(
            r.xrange('metadata')[0][1][b'startTime'],
            dtype=np.float64).item()
        
        # TODO - implement stub test
        # ignore Redis timestamps
        trial_info = tuple(zip(*r.xrange("trial_info")))[1]
        task_state = tuple(zip(*r.xrange("task_state")))[1]
        
        trial_dict = {}
        for trial in trial_info:
            trial_id = int(trial[b'trialNum'])
            trial_start_ts = np.frombuffer(trial[b'trialStart'], dtype=np.float64).item()
            trial_start_time = trial_start_ts - session_start_time
            trial_end_ts = np.frombuffer(trial[b'trialEnd'], dtype=np.float64).item()
            trial_end_time = trial_end_ts - session_start_time
            delay = float(trial[b'delay'])
            intertrial_interval = float(trial[b'interTrialSleep'])
            trial_dict[trial_id] = {
                "start_time": trial_start_time,
                "stop_time": trial_end_time,
                "delay": delay,
                "intertrial_interval": intertrial_interval
            }
        
        event_names = { # TODO - reconsider names
            # b'0': "delay_time", # pretty much redundant with "start_time" (~0.03 ms later) 
            b'1': "go_cue_time",
            # b'3': "task_end_time", # pretty much redundant with "stop_time" (~0.4-0.5 ms earlier)
        }
        for event in task_state:
            trial_id = int(event[b'trialNum'])
            name = event_names.get(event[b'taskState'], None)
            if name:
                timestamp = np.frombuffer(event[b'timeStamp'], dtype=np.float64).item()
                event_time = timestamp - session_start_time
                trial_dict[trial_id][name] = event_time
        
        trial_id_list = sorted(trial_dict.keys())
        trial_list = [trial_dict[i] for i in trial_id_list]
        
        nwbfile.add_trial_column(
            name="delay", 
            description="Length of delay period before go cue, in ms"
        )
        nwbfile.add_trial_column( # TODO - not sure if description accurate
            name="intertrial_interval", 
            description="Length of interval between end of current trial and start of next trial"
        )
        # nwbfile.add_trial_column(
        #     name="delay_time", 
        #     description="Time at which delay period begins"
        # )
        nwbfile.add_trial_column( # TODO - not sure if description accurate
            name="go_cue_time", 
            description="Time of presentation of go cue, which signals beginning of task"
        )
        # nwbfile.add_trial_column( # TODO - not sure if description accurate
        #     name="task_end_time", 
        #     description="Time of the end of the task, when imagined speech and decoding stop"
        # )
        
        for trial in trial_list:
            nwbfile.add_trial(**trial)
        
        r.close()
