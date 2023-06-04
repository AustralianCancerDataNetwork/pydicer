# PyDicer: PYthon Dicom Image ConvertER

Welcome to pydicer, a tool to ease the process of converting DICOM data objects into a format typically used for research purposes. In addition to data conversion, functionality is provided to help analyse the data such as the computing radiomic features or radiotherapy dose metrics. PyDicer uses the NIfTI format to store data is a well defined file system structure. Tracking of these data objects in CSV files, also stored on the file system, provides an easy and flexible way to work with the converted data in your research.

## Installation

PyDicer currently supports Python version 3.8, 3.9 and 3.10. Install PyDicer in your Python
environment using `pip`:

```bash
pip install pydicer
```

## Directory Structure

pydicer will place converted and intermediate files into a specific directory structure. Within the configured working directory `[working]`, the following directories will be generated:

- `[working]/data`: Directory in which converted data will be placed
- `[working]/quarantine`: Files which couldn't be preprocessed or converted will be placed in here for you to investigate further
- `[working]/.pydicer`: Intermediate files as well as log output will be stored in here
- `[working]/[dataset_name]`: Clean datasets prepared using the Dataset Preparation Module will be stored in a directory with their name and will symbolically link to converted in the `[working]/data` directory

## Pipeline

The pipeline handles fetching of the DICOM data to conversion and preparation of your research dataset. Here are the key steps of the pipeline:

1. **Input**: various classes are provided to fetch DICOM files from the file system, DICOM PACS, TCIA or Orthanc. A TestInput class is also provided to supply test data for development/testing.

2. **Preprocess**: The DICOM files are sorted and linked. Error checking is performed and resolved where possible.

3. **Conversion**: The DICOM files are converted to the target format (NIfTI).

4. **Visualistion**: Visualistions of data converted are prepared to assist with data selection.

5. **Dataset Preparation**: The appropriate files from the converted data are selected to prepare a clean dataset ready for use in your research project!

6. **Analysis**: Radiomics and Dose Metrics are computed on the converted data.

## Getting Started

Running the pipeline is easy. The following script will get you started:

```python
from pathlib import Path

from pydicer.input.test import TestInput
from pydicer import PyDicer

# Configure working directory
directory = Path("./testdata")
directory.mkdir(exist_ok=True, parents=True)

# Fetch some test DICOM data to convert
dicom_directory = directory.joinpath("dicom")
dicom_directory.mkdir(exist_ok=True, parents=True)
test_input = TestInput(dicom_directory)

# Create the PyDicer tool object and add the dicom directory as an input location
pydicer = PyDicer(directory)
pydicer.add_input(dicom_directory)

# Run the pipeline
pydicer.run_pipeline()
```

or check out the [Getting Started Example](https://australiancancerdatanetwork.github.io/pydicer/_examples/GettingStarted.html).
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/australiancancerdatanetwork/pydicer/blob/main/examples/GettingStarted.ipynb)
