import sys
import redis
import numpy as np
from typing import Union, Literal, Optional
from hdmf.data_utils import GenericDataChunkIterator


def safe_decode(string_or_bytes: Union[str, bytes], encoding="utf-8"):
    """Convenience method to decode var that can be either str or bytes"""
    if isinstance(string_or_bytes, str):
        return string_or_bytes
    else:
        return str(string_or_bytes, encoding=encoding)


def read_entry(
    entry: dict,
    field: Union[bytes, str],
    encoding: Literal["buffer", "string"],
    dtype: Optional[Union[str, type, np.dtype]] = None,
    shape: Optional[tuple] = None,
    index: Optional[int] = None,
    transpose: bool = False,
):
    """Reads a given field from a redis stream entry"""
    if isinstance(field, str):
        field = bytes(field, "utf-8")
    raw_data = entry[field]
    if encoding == "string":
        if dtype is None:
            data = str(raw_data, "utf-8")
        else:
            try:
                data = np.dtype(dtype).type(safe_decode(raw_data, "utf-8"))
            except Exception as e:
                raise ValueError(f"Unable to cast {safe_decode(raw_data, 'utf-8')} to {dtype}: {e}")
    elif encoding == "buffer":
        data = np.frombuffer(raw_data, dtype=dtype)
        if shape is not None:
            data = data.reshape(*shape)
        if transpose:
            data = data.T
        if index is not None:
            data = data[[index]]
    return data


def read_stream_fields(
    client: redis.Redis,
    stream_name: str,
    field_kwargs: dict[str, dict] = {},
    return_ids: bool = True,
    chunk_size: Optional[int] = None,
    max_stream_len: Optional[int] = None,
    min_id: Optional[bytes] = None,
    max_id: Optional[bytes] = None,
):
    """Reads a set of fields from a Redis stream"""
    # check if no data requested
    if not return_ids and not field_kwargs:
        return {}
    # build data storage dict
    field_data = {key: [] for key in field_kwargs.keys()}
    if return_ids:
        field_data.update({"ids": []})
    # start loop
    max_stream_len = max_stream_len or np.inf
    min_id = min_id or "-"
    max_id = max_id or "+"
    field = list(field_data.keys())[0]
    stream_entries = client.xrange(stream_name, min=min_id, max=max_id, count=chunk_size)
    while len(stream_entries) > 0 and len(field_data[field]) < max_stream_len:
        # get ids if desired
        if return_ids:
            field_data["ids"] += [entry[0] for entry in stream_entries]
        # read fields from each entry
        for entry in stream_entries:
            for field, kwargs in field_kwargs.items():
                field_data[field].append(read_entry(entry=entry[1], field=field, **kwargs))
        # read next entries
        # '(' notation means exclusive range: see https://redis.io/commands/xrange/
        stream_entries = client.xrange(stream_name, min=b"(" + stream_entries[-1][0], max=max_id, count=chunk_size)
    # return
    return field_data


class RedisDataChunkIterator(GenericDataChunkIterator):
    """DataChunkIterator for Redis stream data"""

    def __init__(
        self,
        client: redis.Redis,
        stream_name: str,
        field: str,
        entry_ids: list[bytes],
        read_kwargs: dict = {},
        buffer_gb: float = 0.2,
        chunk_mb: float = 10.0,
        display_progress: bool = False,
        progress_bar_options: Optional[dict] = None,
        max_stream_len: Optional[int] = None,
    ):
        # arg checks
        assert read_kwargs.get("dtype") is not None
        if max_stream_len is not None:
            assert max_stream_len > 0
        # save args
        self.client = client
        self.stream_name = stream_name
        self.field = field
        self.entry_ids = entry_ids
        self.read_kwargs = read_kwargs
        self.max_stream_len = max_stream_len
        # manually compute chunk and buffer shape
        chunk_shape, buffer_shape, frames_per_entry = self._get_chunk_buffer_shape(chunk_mb, buffer_gb)
        self.frames_per_entry = frames_per_entry
        # init parent class
        super().__init__(
            buffer_gb=None,
            buffer_shape=buffer_shape,
            chunk_mb=None,
            chunk_shape=chunk_shape,
            display_progress=display_progress,
            progress_bar_options=progress_bar_options,
        )

    def _get_chunk_buffer_shape(self, chunk_mb, buffer_gb):
        # convert sizes to bytes
        chunk_bytes = chunk_mb * 1e6
        buffer_bytes = buffer_gb * 1e9
        # get size of single entry
        entry = self.client.xrange(self.stream_name, count=1)[0]
        entry_bytes = (
            sys.getsizeof(entry[0]) + sys.getsizeof(entry[1]) + sum([sys.getsizeof(v) for v in entry[1].values()])
        )
        # get entries per chunk and buffer
        entries_per_chunk = max(chunk_bytes // entry_bytes, 1)  # read at least 1 entry
        entries_per_buffer = (buffer_bytes // (entries_per_chunk * entry_bytes)) * entries_per_chunk
        if self.max_stream_len is not None:
            entries_per_chunk = min(entries_per_chunk, self.max_stream_len)
            entries_per_buffer = min(entries_per_buffer, self.max_stream_len)
        # get chunk and buffer shapes
        data = read_entry(entry=entry[1], field=self.field, **self.read_kwargs)
        chunk_shape = (entries_per_chunk * data.shape[0], data.shape[1])
        buffer_shape = (entries_per_buffer * data.shape[0], data.shape[1])
        return chunk_shape, buffer_shape, data.shape[0]

    def _get_data(self, selection: tuple[slice]):
        # get entry idx range to read
        start_idx = selection[0].start // self.frames_per_entry
        end_idx = (selection[0].stop - 1) // self.frames_per_entry  # xrange max is inclusive
        # convert to entry id
        start_id = self.entry_ids[start_idx]
        end_id = self.entry_ids[end_idx]
        # read data from redis
        entries = self.client.xrange(self.stream_name, min=start_id, max=end_id)
        data = np.concatenate(
            [read_entry(entry=entry[1], field=self.field, **self.read_kwargs) for entry in entries], axis=0
        )
        # extra slice if necessary
        if (selection[0].start % self.frames_per_entry) != 0 or (selection[0].stop % self.frames_per_entry) != 0:
            start_offset = selection[0].start % self.frames_per_entry
            data = data[start_offset : (start_offset + selection[0].stop - selection[0].start)]
        return data

    def _get_dtype(self):
        return np.dtype(self.read_kwargs.get("dtype"))

    def _get_maxshape(self):
        entry = self.client.xrange(self.stream_name, count=1)[0][1]
        data = read_entry(entry=entry, field=self.field, **self.read_kwargs)
        assert len(data.shape) == 2
        stream_len = self.max_stream_len or self.client.xlen(self.stream_name)
        return (stream_len * data.shape[0], data.shape[1])
