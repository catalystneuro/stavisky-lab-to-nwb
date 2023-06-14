"""Class for converting generic ecephys data."""
from pynwb.file import NWBFile

from neuroconv.basedatainterface import BaseDataInterface

class StaviskySpikingBandPowerInterface(BaseDataInterface):
    """Spiking band power interface for Stavisky Redis conversion"""
    
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
        chunk_size: int = 1000,
    ):
        # Instantiate Redis client and check connection
        r = redis.Redis(
            port=self.source_data["port"],
            host=self.source_data["host"],
        )
        r.ping()

        module_name = "Processed ecephys"
        module_description = "Contains processed ecephys data like spiking band power"
        processing_module = get_module(nwbfile=nwbfile, name=module_name, description=module_description)
        
        # extract 1 ms data
        sbp_1ms = []
        timestamps_1ms = []
        
        stream_entries = r.xrange('neuralFeatures_1ms', count=chunk_size)
        while len(stream_entries) > 0:
            for entry in stream_entries:
                entry_sbp = np.frombuffer(entry[1][b'spiking_band_power'], dtype=np.float32)
                # TODO: use BRAND_time timestamps?
                sbp_1ms.append(entry_sbp)
                timestamps.append(float(str(entry[0], "UTF-8").split("-")[0]))
            stream_entries = r.xrange('neuralFeatures_1ms', min=stream_entries[-1][0], count=chunk_size)
        sbp_1ms = np.stack(sbp_1ms, axis=0)
        timestamps_1ms = np.array(timestamps_1ms)
        
        # extract 20 ms re-binned data also (at their request)
        sbp_20ms = []
        timestamps_20ms = []
        
        stream_entries = r.xrange('binnedFeatures_20ms', count=chunk_size)
        while len(stream_entries) > 0:
            for entry in stream_entries:
                entry_sbp = np.frombuffer(entry[1][b'spiking_band_power_bin'], dtype=np.float32)
                # TODO: use BRAND_time timestamps?
                sbp_20ms.append(entry_sbp)
                timestamps.append(float(str(entry[0], "UTF-8").split("-")[0]))
            stream_entries = r.xrange('binnedFeatures_20ms', min=stream_entries[-1][0], count=chunk_size)
        sbp_20ms = np.stack(sbp_20ms, axis=0)
        timestamps_20ms = np.array(timestamps_20ms)
        
        # TODO: use timestamp smoothing?
        
        sbp_1ms_timeseries = TimeSeries(
            name="spiking_band_power_1ms",
            data=sbp_1ms,
            unit="V^2/Hz", # does this work?
            timestamps=timestamps_1ms,
            description="Spiking band power in the 250 Hz to 5 kHz frequency range computed 1 kHz",
        )
        sbp_20ms_timeseries = TimeSeries(
            name="spiking_band_power_20ms",
            data=sbp_20ms,
            unit="V^2/Hz", # does this work?
            timestamps=timestamps_20ms,
            description="Spiking band power in the 250 Hz to 5 kHz frequency range computed 1 kHz and re-binned to 50 Hz",
        )
        
        processing_module.add_data_interface(logits_timeseries)
        
        r.close()
        
        return nwbfile
        
        