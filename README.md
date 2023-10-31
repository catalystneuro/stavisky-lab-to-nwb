# stavisky-lab-to-nwb
NWB conversion scripts for Stavisky lab data to the [Neurodata Without Borders](https://nwb-overview.readthedocs.io/) data format.


## Installation
## Basic installation

You can install the latest release of the package with pip:

```
pip install stavisky-lab-to-nwb
```

We recommend that you install the package inside a [virtual environment](https://docs.python.org/3/tutorial/venv.html). A simple way of doing this is to use a [conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/concepts/environments.html) from the `conda` package manager ([installation instructions](https://docs.conda.io/en/latest/miniconda.html)). Detailed instructions on how to use conda environments can be found in their [documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).

### Running a specific conversion
Once you have installed the package with pip, you can run any of the conversion scripts in a notebook or a python file:

https://github.com/catalystneuro/stavisky-lab-to-nwb//tree/main/src/simulated_data/simulated_data_conversion_script.py




## Installation from Github
Another option is to install the package directly from Github. This option has the advantage that the source code can be modifed if you need to amend some of the code we originally provided to adapt to future experimental differences. To install the conversion from GitHub you will need to use `git` ([installation instructions](https://github.com/git-guides/install-git)). We also recommend the installation of `conda` ([installation instructions](https://docs.conda.io/en/latest/miniconda.html)) as it contains all the required machinery in a single and simple instal

From a terminal (note that conda should install one in your system) you can do the following:

```
git clone https://github.com/catalystneuro/stavisky-lab-to-nwb
cd stavisky-lab-to-nwb
conda env create --file make_env.yml
conda activate stavisky-lab-to-nwb-env
```

This creates a [conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/concepts/environments.html) which isolates the conversion code from your system libraries.  We recommend that you run all your conversion related tasks and analysis from the created environment in order to minimize issues related to package dependencies.

Alternatively, if you want to avoid conda altogether (for example if you use another virtual environment tool) you can install the repository with the following commands using only pip:

```
git clone https://github.com/catalystneuro/stavisky-lab-to-nwb
cd stavisky-lab-to-nwb
pip install -e .
```

Note:
both of the methods above install the repository in [editable mode](https://pip.pypa.io/en/stable/cli/pip_install/#editable-installs).

### Running a specific conversion
To run a specific conversion, you might need to install first some conversion specific dependencies that are located in each conversion directory:
```
pip install -r src/stavisky_lab_to_nwb/simulated_data/simulated_data_requirements.txt
```

You can run a specific conversion with the following command:
```
python src/stavisky_lab_to_nwb/simulated_data/simulated_data_conversion_script.py
```

## Repository structure
Each conversion is organized in a directory of its own in the `src` directory:

    stavisky-lab-to-nwb/
    ├── LICENSE
    ├── make_env.yml
    ├── pyproject.toml
    ├── README.md
    ├── requirements.txt
    ├── setup.py
    └── src
        ├── stavisky_lab_to_nwb
        │   ├── conversion_directory_1
        │   └── simulated_data
        │       ├── simulated_databehaviorinterface.py
        │       ├── simulated_data_convert_session.py
        │       ├── simulated_data_metadata.yml
        │       ├── simulated_datanwbconverter.py
        │       ├── simulated_data_requirements.txt
        │       ├── simulated_data_notes.md

        │       └── __init__.py
        │   ├── conversion_directory_b

        └── __init__.py

 For example, for the conversion `simulated_data` you can find a directory located in `src/stavisky-lab-to-nwb/simulated_data`. Inside each conversion directory you can find the following files:

* `simulated_data_convert_sesion.py`: this script defines the function to convert one full session of the conversion.
* `simulated_data_requirements.txt`: dependencies specific to this conversion.
* `simulated_data_metadata.yml`: metadata in yaml format for this specific conversion.
* `simulated_databehaviorinterface.py`: the behavior interface. Usually ad-hoc for each conversion.
* `simulated_datanwbconverter.py`: the place where the `NWBConverter` class is defined.
* `simulated_data_notes.md`: notes and comments concerning this specific conversion.

The directory might contain other files that are necessary for the conversion but those are the central ones.

## Interactive data visualizations

The directory `src/stavisky_lab_to_nwb/widgets/` contains custom widgets for visualizing data in the converted NWB files. To use these widgets, you will need to install the additional packages listed in `src/stavisky_lab_to_nwb/widgets/widgets_requirements.txt`. Example code for using the widgets can be found in the `notebooks/` directory.

### Brain-to-text
#### Decoding performance across sessions

`DecodingErrorWidget` computes and displays the word error rates (WER) and phoneme error rates (PER) for each session that is provided to it. You can view the performance across sessions in the `Overview` panel and a breakdown of performance per trial in the `Session Results` panel.

https://github.com/catalystneuro/stavisky-lab-to-nwb/assets/64850082/93d11cc8-7280-4bf7-86aa-302ea3b2af11

#### Trial alignment for processed electrophysiology data

While `nwbwidgets` supports trial alignment and averaging for spiking data, it does not for generic TimeSeries, so the `AlignedAveragedTimeSeriesWidget` offers this functionality, likely most useful for the various processed electrophysiology data computed during the experiment.

https://github.com/catalystneuro/stavisky-lab-to-nwb/assets/64850082/75aa160d-c054-4acd-a99e-2424c5e41792

#### Decoder RNN and language model predictions over time

The `DecodingOutputWidget` displays the phoneme probabilities and the words predicted by the decoder RNN and language model at each step.

https://github.com/catalystneuro/stavisky-lab-to-nwb/assets/64850082/acb5d835-2902-4f3f-9741-1dcc1f323324
