"Redis sorting extractor."
import redis
import numpy as np
import pynwb
from typing import Union, Optional, List, Tuple, Sequence, Literal

from spikeinterface.core import BaseSorting, BaseSortingSegment
from stavisky_lab_to_nwb.redis_interfaces.redisextractormixin import RedisExtractorMixin


class RedisStreamSortingExtractor(BaseSorting, RedisExtractorMixin):
    def __init__(
        self, 
        port: int,
        host: str,
        stream_name: str,
        data_key: Union[bytes, str],
        unit_count: int,
        dtype: Union[str, type, np.dtype],
        unit_ids: Optional[Sequence] = None,
        frames_per_entry: int = 1,
        timestamps: Optional[np.ndarray] = None,
        start_time: Optional[float] = None,
        sampling_frequency: Optional[float] = None,
        timestamp_source: Union[bytes, Literal["redis"]] = "redis",
        timestamp_kwargs: dict = {},
        unit_dim: int = 0,
    ):
        # Instantiate Redis client and check connection
        self._client = redis.Redis(
            port=port,
            host=host,
        )
        self._client.ping()
        
        # Construct unit IDs if not provided
        if unit_ids is None:
            unit_ids = np.arange(unit_count, dtype=int).tolist()
        
        # timestamp kwargs
        default_ts_kwargs = { # TODO: is this a good way to do things?
            "timestamp_unit": "ms",
            "chunk_size": 1000,
            "smoothing_window": 1000,
            "smoothing_stride": 1,
        }
        default_ts_kwargs.update(timestamp_kwargs)
        
        # get entry IDs and timestamps
        stream_len = self._client.xlen(stream_name)
        assert stream_len > 0, "Stream has length 0"
        start_time, sampling_frequency, timestamps, entry_ids = self.get_ids_and_timestamps(
            stream_name=stream_name,
            frames_per_entry=frames_per_entry,
            timestamp_source=timestamp_source,
            start_time=start_time,
            sampling_frequency=sampling_frequency,
            **default_ts_kwargs
        )
        
        # Initialize Sorting and SortingSegment
        # NOTE: does not support multiple segments, assumes continuous recording for whole stream
        BaseSorting.__init__(self, unit_ids=unit_ids, sampling_frequency=sampling_frequency)
        sorting_segment = RedisStreamSortingSegment(
            client=self._client,
            stream_name=stream_name,
            data_key=data_key,
            unit_count=unit_count,
            unit_ids=unit_ids,
            dtype=dtype,
            frames_per_entry=frames_per_entry,
            start_time=start_time,
            # sampling_frequency=sampling_frequency,
            timestamps=timestamps,
            entry_ids=entry_ids,
            unit_dim=unit_dim,
        )
        self.add_sorting_segment(sorting_segment)
        
        # Not sure what this is for?
        self._kwargs = {
            "port": port,
            "host": host,
            "stream_name": stream_name,
            "data_key": data_key,
            "unit_count": unit_count,
            "dtype": dtype,
            "frames_per_entry": frames_per_entry,
            "timestamp_source": timestamp_source,
            # "timestamp_kwargs": timestamp_kwargs,
        }
    
    def get_multiunit_spike_train(
        self,
        unit_ids: list,
        segment_index: Union[int, None] = None,
        start_frame: Union[int, None] = None,
        end_frame: Union[int, None] = None,
        return_times: bool = False,
    ):
        segment_index = self._check_segment_index(segment_index)
        segment = self._sorting_segments[segment_index]
        spike_frames, spike_labels = segment.get_multiunit_spike_train(
            unit_ids=unit_ids, start_frame=start_frame, end_frame=end_frame
        )
        spike_frames = spike_frames.astype('int64', copy=False)
        if return_times:
            if self.has_recording():
                times = self.get_times(segment_index=segment_index)
                return times[spike_frames]
            else:
                t_start = segment._t_start if segment._t_start is not None else 0
                spike_times = spike_frames / self.get_sampling_frequency()
                return t_start + spike_times
        else:
            return spike_frames, spike_labels
        
    def get_all_spike_trains(
        self,
        outputs: Literal["unit_id", "unit_index"] = "unit_id",
    ):
        assert outputs in ("unit_id", "unit_index")
        spikes = []
        for segment_index in range(self.get_num_segments()):
            spike_times, spike_labels = self.get_multiunit_spike_train(unit_ids=self.unit_ids, segment_index=segment_index)
            if outputs == "unit_id":
                id_map = {idx: unit_id for idx, unit_id in enumerate(self.unit_ids)}
                apply_map = lambda idx: id_map[idx]
                spike_labels = np.vectorize(apply_map)(spike_labels)

            spikes.append((spike_times, spike_labels))
        return spikes


class RedisStreamSortingSegment(BaseSortingSegment):
    def __init__(
        self, 
        client: redis.Redis,
        stream_name: str,
        data_key: Union[bytes, str],
        unit_ids: list,
        unit_count: int,
        dtype: Union[str, type, np.dtype],
        timestamps: np.ndarray,
        entry_ids: list[bytes],
        frames_per_entry: int = 1,
        start_time: Optional[float] = None,
        unit_dim: int = 0,
    ):
        BaseSortingSegment.__init__(self, t_start=start_time)
        
        # Assign Redis client and check connection
        self._client = client
        self._client.ping()
        
        # save some variables
        self._stream_name = stream_name
        self._data_key = data_key
        self._unit_count = unit_count
        self._unit_ids = unit_ids
        self._unit_dim = unit_dim
        self._dtype = dtype
        self._entry_ids = entry_ids
        self._frames_per_entry = frames_per_entry
        self._num_samples = frames_per_entry * len(entry_ids)
        self._timestamps = timestamps

    def get_unit_spike_train(
        self,
        unit_id,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
    ) -> np.ndarray:
        # get unit idx
        unit_idx = self._unit_ids.index(unit_id)
        
        # handle None args
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = self._num_samples - 1 # inclusive
        
        # arg check (not allowing negative indices currently)
        assert start_frame >= 0
        assert end_frame < self._num_samples
        
        # convert to entry number and within-entry idx
        start_entry_idx = start_frame // self._frames_per_entry
        # start_frame_idx = start_frame % self._frames_per_entry
        end_entry_idx = end_frame // self._frames_per_entry
        # end_frame_idx = end_frame % self._frames_per_entry
        
        # read needed entries
        stream_entries = self._client.xrange(
            self._stream_name,
            min=self._entry_ids[start_entry_idx],
            max=self._entry_ids[end_entry_idx],
        )
        
        # loop through, convert to numpy and stack
        base_idx = start_entry_idx * self._frames_per_entry
        spike_frames = []
        for n, entry in enumerate(stream_entries):
            entry_data = np.frombuffer(entry[1][self._data_key], dtype=self._dtype)
            assert entry_data.size == (self._frames_per_entry * self._unit_count)
            if self._frames_per_entry > 1:
                if self._unit_dim == 0:
                    entry_data = entry_data.reshape((self._unit_count, self._frames_per_entry)).T
                elif self._unit_dim == 1:
                    entry_data = entry_data.reshape((self._frames_per_entry, self._unit_count))
            else:
                entry_data = entry_data[None, :]
            entry_data = entry_data[:, unit_idx]
            has_spike = np.nonzero(entry_data)[0]
            spike_times = np.repeat(has_spike, entry_data[has_spike]) + base_idx + n * self._frames_per_entry
            spike_frames.append(spike_times)
        spike_frames = np.concatenate(spike_frames, axis=0).astype(int, copy=False)
        
        if (start_frame > 0) or (end_frame < (self._num_samples - 1)):
            spike_frames = spike_frames[(spike_frames >= start_frame) & (spike_frames < end_frame)]
        
        return spike_frames

    def get_multiunit_spike_train(
        self,
        unit_ids: list,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
    ) -> np.ndarray:
        # get unit idxs
        unit_idx = [self._unit_ids.index(ui) for ui in unit_ids]
        
        # handle None args
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = self._num_samples - 1 # inclusive
        
        # arg check (not allowing negative indices currently)
        assert start_frame >= 0
        assert end_frame < self._num_samples
        
        # convert to entry number and within-entry idx
        start_entry_idx = start_frame // self._frames_per_entry
        # start_frame_idx = start_frame % self._frames_per_entry
        end_entry_idx = end_frame // self._frames_per_entry
        # end_frame_idx = end_frame % self._frames_per_entry
        
        # read needed entries
        stream_entries = self._client.xrange(
            self._stream_name,
            min=self._entry_ids[start_entry_idx],
            max=self._entry_ids[end_entry_idx],
        )
        
        # loop through, convert to numpy and stack
        base_idx = start_entry_idx * self._frames_per_entry
        spike_frames = []
        spike_labels = []
        for n, entry in enumerate(stream_entries):
            entry_data = np.frombuffer(entry[1][self._data_key], dtype=self._dtype)
            assert entry_data.size == (self._frames_per_entry * self._unit_count)
            if self._frames_per_entry > 1:
                if self._unit_dim == 0:
                    entry_data = entry_data.reshape((self._unit_count, self._frames_per_entry)).T
                elif self._unit_dim == 1:
                    entry_data = entry_data.reshape((self._frames_per_entry, self._unit_count))
            else:
                entry_data = entry_data[None, :]
            entry_data = entry_data[:, unit_idx]
            stime, scolumn = np.nonzero(entry_data)
            spike_times = np.repeat(stime, entry_data[stime, scolumn]) + base_idx + n * self._frames_per_entry
            spike_units = np.repeat(scolumn, entry_data[stime, scolumn])
            spike_frames.append(spike_times)
            spike_labels.append(spike_units)
        spike_frames = np.concatenate(spike_frames, axis=0).astype(int, copy=False)
        spike_labels = np.concatenate(spike_labels, axis=0).astype(int, copy=False)
        
        if (start_frame > 0) or (end_frame < (self._num_samples - 1)):
            spike_frames = spike_frames[(spike_frames >= start_frame) & (spike_frames < end_frame)]
            spike_labels = spike_labels[(spike_frames >= start_frame) & (spike_frames < end_frame)]
        
        return spike_frames, spike_labels
    

def redis_add_sorting(
    sorting: BaseSorting,
    nwbfile: Optional[pynwb.NWBFile] = None,
    unit_ids: Optional[List[Union[str, int]]] = None,
    property_descriptions: Optional[dict] = None,
    skip_properties: Optional[List[str]] = None,
    skip_features: Optional[List[str]] = None,
    write_as: Literal["units", "processing"] = "units",
    units_name: str = "units",
    units_description: str = "Autogenerated by neuroconv.",
):
    """
    Slightly modified version of neuroconv.tools.spikeinterface.add_sorting that uses a modified 
    add_units_table (described below). Refer to neuroconv.tools.spikeinterface.add_sorting 
    for documentation on arguments.
    """
    assert write_as in [
        "units",
        "processing",
    ], f"Argument write_as ({write_as}) should be one of 'units' or 'processing'!"
    write_in_processing_module = False if write_as == "units" else True

    redis_add_units_table(
        sorting=sorting,
        unit_ids=unit_ids,
        nwbfile=nwbfile,
        property_descriptions=property_descriptions,
        skip_properties=skip_properties,
        skip_features=skip_features,
        write_in_processing_module=write_in_processing_module,
        units_table_name=units_name,
        unit_table_description=units_description,
        write_waveforms=False,
    )


def redis_add_units_table(
    sorting: BaseSorting,
    nwbfile: pynwb.NWBFile,
    unit_ids: Optional[List[Union[str, int]]] = None,
    property_descriptions: Optional[dict] = None,
    skip_properties: Optional[List[str]] = None,
    skip_features: Optional[List[str]] = None,
    units_table_name: str = "units",
    unit_table_description: str = "Autogenerated by neuroconv.",
    write_in_processing_module: bool = False,
    write_waveforms: bool = False,
    waveform_means: Optional[np.ndarray] = None,
    waveform_sds: Optional[np.ndarray] = None,
    unit_electrode_indices=None,
):
    """
    Slightly modified version of neuroconv.tools.spikeinterface.add_units_table that fetches all
    spike trains at once instead of separately for each unit, as this is more efficient when querying
    Redis. Refer to neuroconv.tools.spikeinterface.add_units_table for documentation on arguments.
    """
    if not write_in_processing_module and units_table_name != "units":
        raise ValueError("When writing to the nwbfile.units table, the name of the table must be 'units'!")

    if not isinstance(nwbfile, pynwb.NWBFile):
        raise TypeError(f"nwbfile type should be an instance of pynwb.NWBFile but got {type(nwbfile)}")

    if write_in_processing_module:
        ecephys_mod = get_module(
            nwbfile=nwbfile,
            name="ecephys",
            description="Intermediate data from extracellular electrophysiology recordings, e.g., LFP.",
        )
        write_table_first_time = units_table_name not in ecephys_mod.data_interfaces
        if write_table_first_time:
            units_table = pynwb.misc.Units(name=units_table_name, description=unit_table_description)
            ecephys_mod.add(units_table)

        units_table = ecephys_mod[units_table_name]
    else:
        write_table_first_time = nwbfile.units is None
        if write_table_first_time:
            nwbfile.units = pynwb.misc.Units(name="units", description=unit_table_description)
        units_table = nwbfile.units

    default_descriptions = dict(
        isi_violation="Quality metric that measures the ISI violation ratio as a proxy for the purity of the unit.",
        firing_rate="Number of spikes per unit of time.",
        template="The extracellular average waveform.",
        max_channel="The recording channel id with the largest amplitude.",
        halfwidth="The full-width half maximum of the negative peak computed on the maximum channel.",
        peak_to_valley="The duration between the negative and the positive peaks computed on the maximum channel.",
        snr="The signal-to-noise ratio of the unit.",
        quality="Quality of the unit as defined by phy (good, mua, noise).",
        spike_amplitude="Average amplitude of peaks detected on the channel.",
        spike_rate="Average rate of peaks detected on the channel.",
        unit_name="Unique reference for each unit.",
    )
    if property_descriptions is None:
        property_descriptions = dict()
    if skip_properties is None:
        skip_properties = list()

    property_descriptions = dict(default_descriptions, **property_descriptions)

    data_to_add = defaultdict(dict)
    sorting_properties = sorting.get_property_keys()
    excluded_properties = list(skip_properties) + ["contact_vector"]
    properties_to_extract = [property for property in sorting_properties if property not in excluded_properties]

    if unit_ids is not None:
        sorting = sorting.select_units(unit_ids=unit_ids)
        if unit_electrode_indices is not None:
            unit_electrode_indices = np.array(unit_electrode_indices)[sorting.ids_to_indices(unit_ids)]
    unit_ids = sorting.unit_ids

    # Extract properties
    for property in properties_to_extract:
        data = sorting.get_property(property)
        if isinstance(data[0], (bool, np.bool_)):
            data = data.astype(str)
        index = isinstance(data[0], (list, np.ndarray, tuple))
        description = property_descriptions.get(property, "No description.")
        data_to_add[property].update(description=description, data=data, index=index)
        if property in ["max_channel", "max_electrode"] and nwbfile.electrodes is not None:
            data_to_add[property].update(table=nwbfile.electrodes)

    # Unit name logic
    if "unit_name" in data_to_add:
        # if 'unit_name' is set as a property, it is used to override default unit_ids (and "id")
        unit_name_array = data_to_add["unit_name"]["data"]
    else:
        unit_name_array = unit_ids.astype("str", copy=False)
        data_to_add["unit_name"].update(description="Unique reference for each unit.", data=unit_name_array)

    units_table_previous_properties = set(units_table.colnames) - {"spike_times"}
    extracted_properties = set(data_to_add)
    properties_to_add_by_rows = units_table_previous_properties | {"id"}
    properties_to_add_by_columns = extracted_properties - properties_to_add_by_rows

    # Find default values for properties / columns already in the table
    type_to_default_value = {list: [], np.ndarray: np.array(np.nan), str: "", Real: np.nan}
    property_to_default_values = {"id": None}
    for property in units_table_previous_properties:
        # Find a matching data type and get the default value
        sample_data = units_table[property].data[0]
        matching_type = next(type for type in type_to_default_value if isinstance(sample_data, type))
        default_value = type_to_default_value[matching_type]
        property_to_default_values.update({property: default_value})

    # Add data by rows excluding the rows with previously added unit names
    unit_names_used_previously = []
    if "unit_name" in units_table_previous_properties:
        unit_names_used_previously = units_table["unit_name"].data
    has_electrodes_column = "electrodes" in units_table.colnames

    properties_with_data = {property for property in properties_to_add_by_rows if "data" in data_to_add[property]}
    rows_in_data = [index for index in range(sorting.get_num_units())]
    if not has_electrodes_column:
        rows_to_add = [index for index in rows_in_data if unit_name_array[index] not in unit_names_used_previously]
    else:
        rows_to_add = []
        for index in rows_in_data:
            if unit_name_array[index] not in unit_names_used_previously:
                rows_to_add.append(index)
            else:
                unit_name = unit_name_array[index]
                previous_electrodes = units_table[np.where(units_table["unit_name"][:] == unit_name)[0]].electrodes
                if list(previous_electrodes.values[0]) != list(unit_electrode_indices[index]):
                    rows_to_add.append(index)
    
    # Extract and concatenate the spike times from multiple segments
    all_spike_times = []
    all_spike_labels = []
    for segment_index in range(sorting.get_num_segments()):
        segment_spike_times, segment_spike_labels = sorting.get_multiunit_spike_train(
            unit_ids=unit_ids, segment_index=segment_index, return_times=True
        )
        all_spike_times.append(segment_spike_times)
        all_spike_labels.append(segment_spike_labels)
    all_spike_times = np.concatenate(all_spike_times, axis=0)
    all_spike_labels = np.concatenate(all_spike_labels, axis=0)

    for row in rows_to_add:
        unit_kwargs = dict(property_to_default_values)
        for property in properties_with_data:
            unit_kwargs[property] = data_to_add[property]["data"][row]
        
        spike_times = all_spike_times[(all_spike_labels == row)]
        if waveform_means is not None:
            unit_kwargs["waveform_mean"] = waveform_means[row]
            if waveform_sds is not None:
                unit_kwargs["waveform_sd"] = waveform_sds[row]
            if unit_electrode_indices is not None:
                unit_kwargs["electrodes"] = unit_electrode_indices[row]
        units_table.add_unit(spike_times=spike_times, **unit_kwargs, enforce_unique_id=True)
    # added_unit_table_ids = units_table.id[-len(rows_to_add) :]  # TODO - this line is unused?

    # Add unit_name as a column and fill previously existing rows with unit_name equal to str(ids)
    previous_table_size = len(units_table.id[:]) - len(unit_name_array)
    if "unit_name" in properties_to_add_by_columns:
        cols_args = data_to_add["unit_name"]
        data = cols_args["data"]

        previous_ids = units_table.id[:previous_table_size]
        default_value = np.array(previous_ids).astype("str")

        extended_data = np.hstack([default_value, data])
        cols_args["data"] = extended_data
        units_table.add_column("unit_name", **cols_args)

    # Build  a channel name to electrode table index map
    table_df = units_table.to_dataframe().reset_index()
    unit_name_to_electrode_index = {
        unit_name: table_df.query(f"unit_name=='{unit_name}'").index[0] for unit_name in unit_name_array
    }

    indexes_for_new_data = [unit_name_to_electrode_index[unit_name] for unit_name in unit_name_array]
    indexes_for_default_values = table_df.index.difference(indexes_for_new_data).values

    # Add properties as columns
    for property in properties_to_add_by_columns - {"unit_name"}:
        cols_args = data_to_add[property]
        data = cols_args["data"]
        if np.issubdtype(data.dtype, np.integer):
            data = data.astype("float")

        # Find first matching data-type
        sample_data = data[0]
        matching_type = next(type for type in type_to_default_value if isinstance(sample_data, type))
        default_value = type_to_default_value[matching_type]

        extended_data = np.empty(shape=len(units_table.id[:]), dtype=data.dtype)
        extended_data[indexes_for_new_data] = data

        extended_data[indexes_for_default_values] = default_value
        # Always store numpy objects as strings
        if np.issubdtype(extended_data.dtype, np.object_):
            extended_data = extended_data.astype("str", copy=False)
        cols_args["data"] = extended_data
        units_table.add_column(property, **cols_args)