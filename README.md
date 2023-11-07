# stavisky-lab-to-nwb
NWB conversion scripts for Stavisky lab data to the [Neurodata Without Borders](https://nwb-overview.readthedocs.io/) data format.


## Installation
The package can be installed from this GitHub repo, which has the advantage that the source code can be modifed if you need to amend some of the code we originally provided to adapt to future experimental differences. To install the conversion from GitHub you will need to use `git` ([installation instructions](https://github.com/git-guides/install-git)). The package also requires Python 3.9 or 3.10. We also recommend the installation of `conda` ([installation instructions](https://docs.conda.io/en/latest/miniconda.html)) as it contains all the required machinery in a single and simple install

From a terminal (note that conda should install one in your system) you can do the following:

```
git clone https://github.com/catalystneuro/stavisky-lab-to-nwb
cd stavisky-lab-to-nwb
conda env create --file make_env.yml
conda activate stavisky-lab-to-nwb-env
```

This creates a [conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/concepts/environments.html) which isolates the conversion code from your system libraries.  We recommend that you run all your conversion related tasks and analysis from the created environment in order to minimize issues related to package dependencies.

Alternatively, if you have Python 3.9 or 3.10 on your machine and you want to avoid conda altogether (for example if you use another virtual environment tool) you can install the repository with the following commands using only pip:

```
git clone https://github.com/catalystneuro/stavisky-lab-to-nwb
cd stavisky-lab-to-nwb
pip install -e .
```

Note:
both of the methods above install the repository in [editable mode](https://pip.pypa.io/en/stable/cli/pip_install/#editable-installs).

## Repository structure
Each conversion is organized in a directory of its own in the `src` directory, with common utility functions and interfaces in `utils/` and `general_interfaces/`, respectively:

    stavisky-lab-to-nwb/
    ├── LICENSE
    ├── make_env.yml
    ├── pyproject.toml
    ├── README.md
    ├── requirements.txt
    ├── setup.py
    ├── notebooks
    │   └── braintotext.ipynb
    └── src
        ├── stavisky_lab_to_nwb
        │   ├── general_interfaces
        │   │   ├── spikeinterface
        │   │   ├── staviskysortinginterface.py
        │   │   ├── staviskyrecordinginterface.py

        │   │   └── __init__.py
        │   ├── utils
        │   │   ├── redis_io.py
        │   │   ├── timestamps.py
        │   │   └── __init__.py
        │   ├── braintotext
        │   │   ├── braintotext_convert_session.py
        │   │   ├── braintotext_conversion.yml
        │   │   ├── braintotext_metadata.yml
        │   │   ├── braintotextnwbconverter.py
        │   │   ├── braintotext_requirements.txt
        │   │   ├── braintotext_notes.md

        │       └── __init__.py
        │   └── widgets
        │       ├── braintotext_widgets.py

        │       └── widgets_requirements.txt

        └── __init__.py

 For example, for the conversion `braintotext` you can find a directory located in `src/stavisky-lab-to-nwb/braintotext`. Inside each conversion directory you can find the following files:

* `braintotext_convert_session.py`: this script defines the function to convert one full session of the conversion.
* `braintotext_conversion.yml`: conversion configuration in yaml format for this specific conversion.
* `braintotext_requirements.txt`: dependencies specific to this conversion.
* `braintotext_metadata.yml`: metadata in yaml format for this specific conversion.
* `braintotextnwbconverter.py`: the place where the `NWBConverter` class is defined.

The directory might contain other files that are necessary for the conversion but those are the central ones.

## Running a specific conversion
To run a specific conversion, you might need to install first some conversion specific dependencies that are located in each conversion directory:
```
pip install -r src/stavisky_lab_to_nwb/braintotext/braintotext_requirements.txt
```

You can then edit the experiment metadata as desired in the file `src/stavisky_lab_to_nwb/braintotext/braintotext_metadata.yaml`.

Finally, with your Redis server running, you can run a specific conversion with the following command:
```
python src/stavisky_lab_to_nwb/braintotext/braintotext_convert_session.py
```

### Brain-to-text conversion

The `braintotext` conversion function can be found in `src/stavisky_lab_to_nwb/braintotext/braintotext_convert_session.py`. The `session_to_nwb` function can be run by running the file directly, as shown above, or by importing the function elsewhere and running it in an alternate script or notebook. The function takes in a number of arguments:

* port: port number of the running Redis server (i.e. 6379)
* host: host name of the running Redis server (i.e. "localhost")
* conversion_config_path: path to a YAML file configuring the conversion (more details below)
* output_dir_path: path to where you want to save the output NWB file
* source_data: overrides for source data configuration
* conversion_options: overrides for conversion option configuration
* stub_test: whether to convert and save only a portion of the data, used only for testing purposes
* exclude_interfaces: list of names of interfaces to exclude from the conversion

User configuration of the conversions primarily involves editing/creating YAML files. The first YAML file, `braintotext_metadata.yml`, containes experimental metadata, like subject age, that should be provided by the experimenter (if appropriate to share). The other YAML file, `braintotext_conversion.yml`, essentially specifies all keyword arguments necessary to instantiate and run the data interfaces used to read, convert, and write data.

Each data interface is defined in `braintotextnwbconverter.py`, where `BrainToTextNWBConverter.data_interface_classes` specifies the class for each interface. Each top-level entry in `braintotext_conversion.yml` corresponds to one of these interfaces, with sub-dictionaries `source_data` for instantiating the class and `conversion_options` for configuring how it converts the data. For each interface, you can find more documentation on the accepted kwargs in their respective files, specifically their `__init__` functions for `source_data` and their `add_to_nwbfile` functions for `conversion_options`.

To change the configuration for `session_to_nwb`, you can modify `braintotext_conversion.yml`, create a new YAML file and point to it with the `conversion_config_path` argument, or override specific kwargs programmatically with the `source_data` and `conversion_options` arguments. To remove certain data interfaces from the conversion, you can again modify/create YAML files to comment out or remove particular interfaces, or you can use the `exclude_interfaces` argument. Note that you do not need to edit `braintotextnwbconverter.py`, since interfaces will simply not be instantiated if no source data is provided for them, even if they are still in `BrainToTextNWBConverter.data_interface_classes`. In general, we recommend using YAML files to modify the conversion.



## Interactive data visualizations

The directory `src/stavisky_lab_to_nwb/widgets/` contains custom widgets for visualizing data in the converted NWB files. To use these widgets, you will need to install the additional packages listed in `src/stavisky_lab_to_nwb/widgets/widgets_requirements.txt`. Example code for using the widgets can be found in the `notebooks/` directory.

### Brain-to-text
#### Decoding performance across sessions

`DecodingErrorWidget` computes and displays the word error rates (WER) and phoneme error rates (PER) for each session that is provided to it. You can view the performance across sessions in the `Overview` panel and a breakdown of performance per trial in the `Session Results` panel.

https://github.com/catalystneuro/stavisky-lab-to-nwb/assets/64850082/99bd8848-88be-4012-b2e7-b450f67e485d

#### Trial alignment for processed electrophysiology data

While `nwbwidgets` supports trial alignment and averaging for spiking data, it does not for generic TimeSeries, so the `AlignedAveragedTimeSeriesWidget` offers this functionality, likely most useful for the various processed electrophysiology data computed during the experiment.

https://github.com/catalystneuro/stavisky-lab-to-nwb/assets/64850082/75aa160d-c054-4acd-a99e-2424c5e41792

#### Decoder RNN and language model predictions over time

The `DecodingOutputWidget` displays the predictions of the decoder RNN and language model at each timestep. The lower plot shows the predicted phoneme probabilities of the RNN after a softmax activation, with labels for the highest-probability phoneme at a given timestep. The upper plot shows the words predicted by the language model based on the phoneme probabilities. The words with strikethroughs were discarded from the final prediction, while words without strikethroughs were kept to form the full predicted sentence.

https://github.com/catalystneuro/stavisky-lab-to-nwb/assets/64850082/acb5d835-2902-4f3f-9741-1dcc1f323324
