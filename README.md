# Convert vizgen files to starfish format

This repo houses python code for loading and handling vizgen experiment data. Data can be exported to local tiff files, and can also be used to convert raw output of a vizgen experiment to starfish compatible [SpaceTx](https://spacetx-starfish.readthedocs.io/en/latest/index.html) format.

## Installation

To get started, first create a virtual environment from the requirements file. The specific build requires Python 3.8 to meet the requirements of Starfish. Be sure to install and activate the virtual environment before using pip, which installs the `vizgen-tools` command using setuptools to the virtual environment.

```
conda create --file vizgen_tools_environment.yml
conda activate vizgen_tools
pip install .
```

## Usage

At any time, use the `--help` flag to see a list of commands or options available.

```
vizgen-tools --help
```

With the conda environment activated, the export function in the toolkit can be activated by invoking the `export` command.

```
vizgen_tools export
```

Doing so, the CLI tool will prompt you through the required parameters to run the file converstion.

- `experiment-name`: the name of the target experiment
- `raw-dir`: the directory where the vizgen raw data is stored
- `analysis-dir`: the directory where the vizgen analysis data is stored
- `output`: the target directory where to export the final converted data

Note that the tool assumes your input data is structure as bellow:
```
- parent_directory/
    - raw_data/
        - experiment_name/
    - analysis_data/
        - experiment_name/
```

You can also pipe these arguments directly to the command using the parameter flags.

```
python vizgen_tools.py convert
    --to starfish
    --experiment-name experiment_name
    --raw-data /path/to/raw/data/
    --analysis-data /path/to/analysis/data
    --output /path/to/output/dir/
```
For a quick explanation of arguments, use the `--help` flag.

```
vizgen_tools convert --help
```

## Starfish conversion

The tool also contains a command for converting vizgen directly to a starfish format.

```
vizgen-tools convert --to starfish
```

This implementation currently requirings providing a *temporary directory* as an intermediary step in the data conversion (`--tmp /path/to/tmp/dir/`).

- `tmp`: the toolkit requires a temporary path for intermediary data conversion. Please provide a path where:
    - there is sufficient disk space (> size of the original raw data)
    - you have sufficient write permissions

You can use the `--help` flag here as well for more information for running the `convert` command.

```
vizgen-tools convert --help
```
