# PyDicer: Python DICOM Image Converter

## 1. Motivation and Significance

The DICOM image format is ubiquoutous in the medical imaging domain with effectively all imaging data captured by clinical scanners being stored in this format. Supplementary data objects have also been defined in the DICOM standard to store data related to image series such as Radiotherapy treatment planning data.

There are many advantages of using this format within hosptials and clinics however challenges exist when making use of data in this form in research projects. NIfTI

- Challenges with DICOM in research
  - Storing images as series of files rather than just one volume
  - Image orientation can often require adjustment
  - Image units need to be adjusted (e.g. PET)
  - Storing structures as contour points, masks are better suited for common medical image analysis tasks.
  - Dose grids require rescaling
- More suitable image formats (NIfTI)
  - Stores image space parameters as part of format
  - Compatiable with most image analysis tools (such as ITK)
  - Downside: no meta data
- Clinical datasets are messy
  - Research projects often make use of clinical data that has been anonymised for research purposes
  - Manual curation of these datasets is resource intensive and often not feasible for large datasets
  - Even with manual curation there can be errors and additional cleaning is required
- Open-source tool for generalised use across research projects
  - Custom scripts are often prepared per research project converting and storing files into a custom folder structure even if certain tasks are common between projects.
  - A generalised tool removes the need for implementing large amounts of boiler plate code per project.
  - If issues with data conversion are determine these can be solved in the generalised tool for other projects to also benefit

## 2 Software description

### 2.1 Software architecture

PyDicer consists of serveral modules that are pieced together into a suitable processing pipeline for a given project. A working directory is specified in which all data for a project will be stored with PyDicer handling the generation of the directory structure within (see section 2.2). The provides transparency to the user to easily explore data within the directory as well as flexibility as research users will commonly store their dataset in a comparable way. Each module will read the data in the working directory as input and will write to the working directory as its output, like this giving some flexibility as to the order in which the modules are called upon.

Each module provided is described in detail in section 2.3, but a typical pipeline consists of defining an input module which enables fetching of some DICOM data. Next the preprocess and convert modules will convert the data to NIfTI objects within the working directory structure. Once this is done then various other modules can be called upon such as the generate module (to, for example, generate auto-segmentation on images in the dataset) or the analyse module (to compute and extract radiomics or dose metrics). The visualise module and the preparation module can be called upon at various places throughout the pipeline depending on the use case to generate visualisation of data or to extract a subset of data for dataset cleaning prior to analysis.

To help the tool and the user manage the data objects available in the dataset, PyDicer makes use the pandas library. This is used to read and write CSV files to the filesystem which track the data objects created and can be easily filtered when performing image analysis tasks. Since pandas is also popular with researchers working in the data science field this way of tracking data objects provides flexibilty to users using the data in their research.

### 2.2 File system specification

Having a well defined, consistent file system specification is key for the core functionality of PyDicer. This specification was designed in such a way to help ensure that DICOM datasets containing many object per patient are supported. The converted data is written to a sub-folder named **data** within the working directory. Each patient has their own sub-directory within this data directory with the format

- Images
  - [hashed_uid]
    - [modality].nii.gz
    - metadata.json
- Structures
  - [hashed_uid]
    - [structure_name].nii.gz
    - metadata.json
- Plans
  - [hashed_uid]
    - [modality].nii.gz
    - metadata.json
- Dose
  - [hashed_uid]
    - [modality].nii.gz
    - metadata.json
- converted.csv

Like this, each DICOM object converted has it's own sub-folder using 6 character hashed version of the SeriesInstanceUID. This descision was made to make browsing the file system more accessible since the SeriesInstanceUIDs are typically 64 characters and making this browsing diffcult.

### 2.3 Modules

#### 2.3.1 Input

The **Input** module provides DICOM files to PyDicer which comprise the dataset to be processed in the follow steps. This will always be the first step in the pipeline and consists of several sub-modules to choose from. The simplest approach is to copy the DICOM files to a folder on the file system and use the `FileSystemInput` class to provide these to PyDicer.

In clinical scenarios DICOM files are typically stored in a Picture Archiving and Communication Systems (PACS), the `DICOMPACSInput` will fetch DICOM datasets from the PACS specified prior to processing. Simiarly, Orthanc is a popular open-source research tool which provides much of the same functionality as a clinical PACS. Using the `OrthancInput` class data can be extracted from an Orthanc instance. If data is stored in a zip file on the web then then `WebInput` class can be used, or for testing purposed the `TestInput` class downloads some test data.

#### 2.3.2 Preprocess

Before the data can be converted, the preprocess module must first be used to sort through the DICOM data available. This is used to track instances which make up each DICOM series, help identify duplicate files (with the same SeriesInstanceUID), track linkage between files which needed for converting some modalities. This information is store in the `.pydicer/preprocessed.csv` file for use by the following conversion module. If any DICOM files are corrupt or are not of a supported modality then these are copied to the `quarantine` directory.

#### 2.3.3 Convert

This module will use the information from `.pydicer/preprocessed.csv` and convert DICOM object to the appropriate NIfTI representation and store it within the file system specification. This also stores the metadata available in the DICOM headers alongside the converted NIfTI object. If there are any errors while converting any DICOM objects then these are copied to the `quarantine` folder.

#### 2.3.4 Visualise

The visualise module can be called at any stage throughout the pipeline following initial conversion and will prepare cross-section visualisations of the data converted. Several visualisations are prepared for each image, structure set and dose grid. These are stored as a PNG file within the relevant data object directory.

#### 2.3.5 Prepare

To help prepare a subset of data per patient to use for analysis the prepare module can be used. This module provides a significant amount of flexibility as the `prepare_from_dataframe` function can be used by simply applying some filtering to the dataframe containing all data and passing the filtered DataFrame to this function.

Each dataset should have a unique name and a folder will be created within the working directory for the subset of data. This directory structure will exactly match the main converted directory, with the converted.csv files tracking the filtered data objects. However none of the data object folders/files are copied to avoid data duplication. Instead symbolic links can be generated on Unix based file systems to ease the browsing of the data subsets. On Windows files systems the symbolic links cannot be created, however the remaining data subset functionality will coninute to work.

PyDicer provides some out of the functions to filter datasets per patient common in radiotherapy projects. These include `rt_latest_struct` and `rt_latest_dose` which will extract the latest structure set or dose respectively by date along with the data objects linked to those objects.

#### 2.3.6 Generate

As part of medical image analysis projects, it is possible that some additional data objects are generated (which aren't included in the original DICOM data). The `generate` module provides functionality to add these to the PyDicer file structure and ensure they are tracked in the converted objects list. This ensure that other modules (such as visualise and analyse) also run across generated data objects as well as the originally converted objects.

Some common scenarios to generate data objects are producing auto-segmentations on images in the dataset, to apply scaling to dose grids or combination of multiple dose grids.

#### 2.3.7 Analyse

A common task in medical image analysis is to extract radiomic features from data. One popular library to do this is the `pyradiomics`. The analyse module provides functionality to compute the desired radiomic features from pyradiomics across the data available. The linkage between structures and images is done by the analyse module before computing radiomics. These are stored in the structure set directory to which they belong so that they can easily be extracted when needed.

In radiotherapy research another common task is to compute Dose Volume Histograms (DVHs) and to then extract dose metrics from these. The analyse module computes the DVHs and stores these as CSV files in the dose directory to which they belong. Visualisations of dose volume histograms are also stored in PNG format along side the CSV file. Once these are compute the user can use the `compute_dose_metrics` to extract the desired dose metrics from these when needed.

## Illustrative Example

- Pull from PACS
- Preprocess/Convert
- Prepare subset
- Generate auto-segmentation (cardiac)
- Visualise results
- Analyse radiomic and dose metrics

## Impact

- Has been used to process data in:
  - Auto-segmentation feasibility project (Shrikant)
  - AusCAT cardiac toxicities project (Vicky/Rob)
  - Lung SABR clinical trial cardiac toxicities project (VickyC)
  - Automated contour quality assurance project (PhilC)
  - PET analysis (Yuvnik)
  - Gyane MRI Radiomics Analysis (Rhianna)
  - Bone Metastisis auto-segmentation (Iromi)
  - PROMETHOUS Toxicities analysis (Michael Cardoso)

## Conclusions
