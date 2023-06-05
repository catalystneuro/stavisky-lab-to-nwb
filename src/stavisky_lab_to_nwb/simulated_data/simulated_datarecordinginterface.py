import redis
import numpy as np
from pathlib import Path

from pynwb.file import NWBFile

from neuroconv.datainterfaces.ecephys.baserecordingextractorinterface import BaseRecordingExtractorInterface


class SimulatedDataRecordingInterface(BaseRecordingExtractorInterface):
    """Recording interface for simulated_data conversion"""
    
    ExtractorName = "NumpyRecording"

    def __init__(
        self,
        port: int,
        host: str,
        verbose: bool = True,
        es_key: str = "ElectricalSeries",
    ):
        # This should load the data lazily and prepare variables you need
        r = redis.Redis(
            port=port,
            host=host,
        )
        r.ping()
        traces_list, sampling_frequency, t_starts = self.load_data(client=r)
        r.close()
        super().__init__(
            verbose=verbose,
            es_key=es_key,
            traces_list=traces_list,
            sampling_frequency=sampling_frequency,
            t_starts=t_starts,
        )
    
    def load_data(self, client: redis.Redis):
        r = client
        
        sampling_frequency = 30000.
        
        continuous_data = r.xrange('continuousNeural')
        tracking_ids = np.array([
            np.frombuffer(entry[1][b'tracking_id'], dtype=np.int64).item() 
            for entry in continuous_data])
        packet_order = np.argsort(tracking_ids)
        
        voltage_traces = np.concatenate([
            np.frombuffer(continuous_data[i][1][b'samples'], dtype=np.int16).reshape(30, 256)
            for i in packet_order], axis=0)
        
        session_start_time = np.frombuffer(
            r.xrange('metadata')[0][1][b'startTime'],
            dtype=np.float64).item()
        recording_start_time = int(continuous_data[0][0].split(b'-')[0]) / 1000 - 0.001 - session_start_time
        
        return [voltage_traces], sampling_frequency, [recording_start_time]
