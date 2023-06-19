# Notes concerning the Stavisky conversion

## Reading RDB files

Start server from shell with

```
redis-server --dir /path/to/rdb --dbfilename rdbname.rdb --port 6379
```

Then, in Python:

```
import redis
r = redis.Redis(port=6379, host="localhost")
r.ping() # check connection
```

View all streams:
```
r.keys()
```

View stream length:
```
r.xlen()
```

Get all entries from stream:
```
r.xrange(stream_name)
```

Get select entries from stream:
```
r.xrange(stream_name, min=min, max=max, count=count)
```
`min='-'` means earliest available entry, `max='+'` means latest. `count` takes precedence over `min` and `max`, starting from the earliest entry. `min` and `max` must be Unix timestamps with millisecond resolution.

Output of `r.xrange()` is a list of length-2 tuples. Each tuple contains an ID and a dict of data. The IDs are Unix millisecond timestamps. When multiple entries occur in the same millisecond, IDs are appended with `-0`, `-1`, etc. in order. All keys and values are bytestrings (so we must prepend keys with `b'` to access).

Much of the data are likely stored as byte buffers of numpy arrays, which can be read with
```
np.frombuffer(bytestring, dtype=dtype)
```

## Raw data summary

Here is an overview of each data stream in the RDB file: *(bold question marks means need to seek clarification on units, etc.)*

- `"supergraph_stream"` - List of length 1. The singular dict has 1 key, `"data"`, which maps to a massive JSON dict, which has keys:
  - `"redis_host"` - IP address of Redis instance, `192.168.137.30`
  - `"redis_port"` - port number, `6379`
  - `"graph_name"` - name of file / experiment
  - `"graph_loaded_ts"` - probably session start time, possibly Unix timestamp? Ex: `12590134766400`
  - `"nodes"` - big dict of each node in the graph, with its inputs/outputs and some metadata. These are the nodes running on the primary computer. Includes:
    - `"metadata_node"` - participant name, session name, session description
    - `"microphone"` - audio sampling rate, microphone name, etc.
    - `"mfcc"` - sampling rate, frequency ranges, etc.
    - `"spike_generator"`, `"spike_gen_1"`, `"cb_gen_1"`, and `"cerebusAdapter"`  - neuron count, firing rate ranges, sampling rates, random seeds, etc.
    - `"feature_extraction"` - filtering params like order, cutoffs, filter type (butterworth)
    - `"binning20ms"` - bin width
    - `"sentenceTask_commandLine"` - some info about trial generation parameters (delay length ranges)
    - `"sentenceTask_pyglet"` - some GUI info? screen size, etc.
    - `"brainToText_closedLoop"` - decoder setup, like path to models used
    - `"tts_node"` - text-to-speech model path
    - `"derivative_runner"` - derivative processes, here just something that converts RDB to `.mat`
    - `"neural_visualizer"` - mostly redundant info, probably just a GUI for plotting
- `"supervisor_ipstream"` - List of length 3. The 3 commands passed to Redis. Nothing particulary useful here.
- `"booter"` - List of length 2. Each node running on the secondary computer and their parameters. First entry corresponds to command `"startGraph"` and contains what seems to be a copy of data dict in `"supergraph_stream"`. Second entry corresponds to command `"stopGraph"` and is otherwise empty. Nothing particulary useful.
- `"booter_status"` - List of length 1. Status of second computer, nothing particulary useful.
- `"trial_info"` - List of length 40. Information about each trial, obviously. Keys:
  - `"trialNum"` - int indicating trial number (0 - 39)
  - `"trialStart"` - float64 Unix timestamps with sub-millisecond resolution
  - `"trialEnd"` - float64 Unix timestamps with sub-millisecond resolution
  - `"delay"` - int indicating delay from trial start to go cue, in milliseconds
  - `"interTrialSleep"` - intertrial interval between current and next trial, in milliseconds
  - `"sentenceCue"` - text sentence used as audio input/decoding target for the trial
- `"binned:decoderOutput:stream"` - List of length 5750. Output of brain-to-text decoder. Entries follow the pattern \[start, logits, decoded sentence so far, logits, decoded sentence so far, ... , logits, final decoded sentence, end, start, logits ...\]. 
  - start dict - has one key, `"start"`, which maps to corresponding entry ID which is unix timestamp. Ex: `b'1677021306179-0'`
  - logits dict - has one key, `"logits"`, which maps to 164-byte sequence, float32 dtype, array of shape (41,). Each entry corresponds to the 39 phonemes, plus silence and a space between words
  - decoded sentence dict - has one key, either `"partial_decoded_sentence"` or `"final_decoded_sentence"`, mapping to text decoded so far. Ex: `b' my family is very need'`
  - end dict - has one key, `"end"`, which maps to corresponding entry ID which is unix timestamp, like `"start"`
- `"binnedFeatures_20ms"` - List of length 22902. Binned neural features (threshold crossings and spike power, 20 ms bins). These aren’t z-scored yet. Keys:
  - `"threshold_crossings_bin"` - sequence of 512 bytes. int16, array shape (256,)
  - `"spike_band_power_bin"` - sequence of 1024 bytes, float32, array shape (256,)
  - `"input_id"` - 160-byte sequence. int64, list of 20 indices corresponding to `"neuralFeatures_1ms"` included in this bin
  - `"tracking_id"` - 8-byte sequence. int64, \[19, 39, ...\]. Corresponds to `"neuralFeatures_1ms"`
  - `"BRAND_time"` - 8 bytes, int64 of `time.monotonic_ns()`
- `"metadata"` - List of length 1. Dict contains basic info: participant, session_name, session_description, block_num, block_description, startTime
- `"mfcc"` - List of length 91614 (so ~200 Hz?). MFCC features computed from audio. Keys:
  - **????** `"data"` - 52-byte sequence, likely 13 float32s.
  - **????** `"ts"` - string timestamp, presumably in seconds. Ex: `b'12601.819151985'`. Frequency seems to be irregular?
  - `"i"` - index, corresponds to position of ths dict in the list.
- `"microphone"` - List of length 91666 (so ~200 Hz?). Raw microphone data. Keys:
  - `"data"` - 440-byte sequence. Unknown dtype
  - `"ts"` - string timestamp, presumably in seconds. Ex: `b'12601.819151985'`. Assuming these are in seconds, frequency seems to be around 200 Hz. This would suggest 220 values in `"data"` as the audio is 44 kHz. So, dtype is float16 maybe?
  - `"i"` - index, corresponds to position of ths dict in the list.
- `"task_state"` - List of length 120. The current state of the task (0/1/3 for delay/go cue/trial end). 3 entries per trial. Keys:
  - `"trialNum"` - current trial number
  - `"taskState"` - string of int, either 0/1/3 as described above
  - `"timeStamp"` - timestamp in same format is `"trial_info"`, so unknown format. Ex: `b'\x1e\xc6s!T\xfd\xd8A'`
- `"graph_status"` - List of length 9. Status of graph (e.g. initialized, running, etc.). Doesn't seem particularly useful
- `"threshold_values1"` - List of length 458055 (so ~1 kHz?). Spikes simulated at 30 kHz from model-generated firing rates. Keys:
  - **????** `"ts_start"` - string timestamp. Ex: `b'12601.696126469'`
  - **????** `"ts_in"` - string timestamp, same format as above. Not sure what it means. 
  - **????** `"ts"` - another string timestamp, same format as above.
  - **????** `"ts_end"` - another string timestamp.
  - `"i"` - string of int, index in list.
  - **????** `"i_in"` - string of int, appears to be equal to `i // 5`
  - **????** `"continuous"` - sequence of 15360 bytes. Guess? 256 channels, (u)int16 dtype, 30 samples per channel, as these are output at 1 kHz but spikes are simulated at 30 kHz. Unsure of proper array reshaping, though.
  - **????** `"thresholds"` - sequence of 256 bytes. uint8s for each channel, threshold crossings in the last 1 ms?
- `"firing_rates"` - List of length 91611 (so ~200 Hz?). Firing rates simulated from microphone data. Keys:
  - **????** `"rates"` - 1024-byte sequence. Probably 256 channels of float32.
  - `"ts"` - string timestamp, presumably in seconds.
  - `"i"` - index, corresponds to position of ths dict in the list.
- `"cb_gen_1"` - List of length 458055 (so ~1 kHz?). UDP packets of raw neural data simulated from spikes. Keys:
  - **????** `"timestamps"` - 16 bytes. Not sure how to read, but from a quick search on UDP it seems that the data is streamed separately? So no actual neural data stored in RDB?
- `"neuralFeatures_1ms"` - List of length 458055 (so ~1 kHz?). 1 ms binned neural features (like `"binnedFeatures_20ms"`). Keys:
  - `"threshold_crossings"` - sequence of 512 bytes. int16, array shape (256,)
  - `"spike_band_power"` - sequence of 1024 bytes, float32, array shape (256,)
  - `"nsp_timestamps"` - 240-byte sequence. int64, simply indices (at 30 kHz) starting from 1 of entries corresponding to this bin
  - `"tracking_id"` - 8-byte sequence. Appears to be int64, indices starting from 1 (at 1 kHz)
  - `"BRAND_time"` - 8 bytes, int64, `time.monotonic_ns()`
- `"buttonAdapter_output"` - List of length 86. Information about when the button is pressed (for ending each trial). Keys:
  - `"direction"` - `{"down", "up"}` for button press and release (respectively)
  - `"event_timestamp"` - string of float of event time, Unix timestamp (submillisecond)
  - `"time_display"` - clock time (string, `%H:%M:%S`)
  - `"write_timestamp"` - raw bytes float64 of `"event_timestamp"`
- `"continuousNeural"` - List of length 458055 (so ~1 kHz?). 30 kHz simulated neural data through virtual recording interface. Keys:
  - `"timestamps"` - 240-byte sequence. int64, (30,), indices (at 30 kHz) starting from 1.
  - `"BRANDS_time"` - 240-byte sequence. float64, (30,), `time.monotonic_ns() / 1e9`, in seconds
  - `"udp_recv_time"` - 240-byte sequence. float64, (30,) array of Unix timestamps
  - `"tracking_id"` - 8 bytes, int64, index starting from 1 (at 1 kHz)
  - `"write_timestamp"` - float64 Unix timestamp
  - `"samples"` - 15360-byte sequence. 30 kHz data, int16, 0.1 mV scale, use `.reshape(30, 256)`.

## Data to extract

Definitely:
- trial information (`"trial_info"` and `"task_state"`)
- `"continuousNeural"` (continuous voltage trace)
- `"neuralFeatures_1ms"` (threshold crossing times, spike band power)
- `"binnedFeatures_20ms"` (spiking band power only, as threshold crossings can be binned from units. though redundant, storing as requested by lab)
- `"binned:decoderOutput:stream"` (output of speech decoder)

No:
- `"microphone"`, `"mfcc"`. Microphone input and feature extraction are used to generate simulated neural data. So, they probably won't be present in actual experimental data.
- `"threshold_values1"`, `"firing_rates"`. Simulated neural activity parameters. Again, won't be present in actual experiments.
- `"cb_gen_1"`. Doesn't seem to contain any actual data and only records of when/where data was sent through UDP. 
- `"buttonAdapter_output"`. Seems pretty useless, used to control trial timings, etc. but those are stored elsewhere.
