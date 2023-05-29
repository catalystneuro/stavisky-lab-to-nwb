# Notes concerning the simulated_data conversion

## Data summary

Data are stored as lists of (id, dict) in each stream. Here is an overview of each data stream in the RDB file: *(bold question marks means need to seek clarification on units, etc.)*

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
  - **????** `"trialStart"` - raw bytes timestamp, not sure what format? Ex: `b'g\xfd\xcf\x1dT\xfd\xd8A' = 0x67fdcd1d54fdd841`
  - **????** `"trialEnd"` - same as `"trialStart"`. Ex: `b'c\xcds!T\xfd\xd8A' = 0x63cd732154fdd841`
  - **????** `"delay"` - int indicating some sort of delay, but not sure what delay? Potentially in milliseconds? Ex: 3580
  - **????** `"interTrialSleep"` - intertrial interval between current and next trial (or previous and current trial)? in milliseconds? Ex: 2000
  - `"sentenceCue"` - text sentence used as audio input/decoding target for the trial
- `"binned:decoderOutput:stream"` - List of length 5750. Output of brain-to-text decoder. Entries follow the pattern \[start, logits, decoded sentence so far, logits, decoded sentence so far, ... , logits, final decoded sentence, start, logits ...\]. 
  - **????** entry ID for each dict - may be timestamp? Ex: `b'1677021306179-0'`
  - start dict - has one key, `"start"`, which maps to corresponding entry ID which might be a timestamp? Ex: `b'1677021306179-0'`
  - **????** logits dict - has one key, `"logits"`, which maps to byte sequence. Not sure how to read byte sequence. Ex: `b'\x8c\x99\xb0A\x82P\x14A\xc2\x81\xa5\xc0P3\x96\xc0A\xbd5?p\xed\x1e\xc0L\xb2\xbd\xbf\xaf\xf1\xb3\xbfH\xa9d\xbf\xee\xed\xa6\xbf\xf2\x13\xd7\xbf\xce\xe2\xcd?$M.?\xba\xf1\x06\xc0\xeb\x9a\x16\xc0|\x9ft?5\xbf\xb4>*\xf7\xe0\xbfV\x1c\x93?C\x03\x92\xbe\xbb?i\xc0\xf0n\t?\xd1\x06\xce>\xa8\x8f\xa6?\xa2wA?\xaaM\x8b>\xe1\xafK\xbf\xd4c\xa3\xc0\\\xfda@\xa4\xf0\n\xbf\xb8\xe2\xa1@\xb8\xecV\xc0e\xb9s@\xa6\x14\x8a@\xddS\x8b\xc0\xf6\x86\xaf\xbf\xe0\xff\x01\xc0t\xb2&\xbf\x1f\xbc(\xc0\x08\xb7\x80?l\xc3\x8a\xc0'`
  - decoded sentence dict - has one key, either `"partial_decoded_sentence"` or `"final_decoded_sentence"`, mapping to text decoded so far. Ex: `b' my family is very need'`
- `"binnedFeatures_20ms"` - List of length 22902. Binned neural features (threshold crossings and spike power, 20 ms bins). These aren’t z-scored yet. Again, entry IDs may be timestamps, like for `"binned:decoderOutput:stream"`. Keys:
  - **????** `"threshold_crossings_bin"` - sequence of 512 bytes. Possibly 2 bytes per channel. If so, numbers seem to suggest little-endian ordering, simply storing int of threshold crossings in this bin
  - **????** `"spike_band_power_bin"` - sequence of 1024 bytes, so likely 4 bytes per channel. Unsure of format, float 32 maybe?
  - **????** `"input_id"` - 160-byte sequence. ??
  - **????** `"tracking_id"` - 8-byte sequence. ??
  - **????** `"BRAND_time"` - 8 bytes, unknown format. Ex: `b'\x1b\x06\x0e-w\x0b\x00\x00'`
