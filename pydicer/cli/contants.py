def get_sub_help_mesg(input_commands):
    # pylint: disable=missing-function-docstring
    help_mesg = f"""Subcommand of the following: {input_commands}


    test WORKING_DIRECTORY_PATH
        - WORKING_DIRECTORY_PATH (str, optional): The working directory in which to
    store the data fetched. Defaults to a temp directory.
    
        Example usage:
            python -m pydicer.cli.run input --type test cli_test
    

    pacs HOST_IP PORT AE_TITLE WORKING_DIRECTORY_PATH MODALITY [PATIENT_IDs]
        - HOST_IP (str, optional): The IP address of host name of DICOM PACS. Defaults to 
    'www.dicomserver.co.uk'.
        - PORT (int, optional): The port to use to communicate on. Defaults to 11112.
        - AE_TITLE (str, optional): AE Title to provide the DICOM service. Defaults to 
    None.
        - WORKING_DIRECTORY_PATH (str, optional): The working directory in which to
    store the data fetched. Defaults to a temp directory.
        - MODALITY (str, optional): The modality to retrieve DICOMs for. Defaults 
    to 'GM'.
        - PATIENT_IDs (str, required): a string-list of patient IDs (IDs seperated by spaces)
     to retrieve the DICOMs for.
    
        Example usage:
            python -m pydicer.cli.run input --type pacs www.dicomserver.co.uk 11112 DCMQUERY 
    cli_test GM PAT004 PAT005
    
    
    web DATA_URL WORKING_DIRECTORY_PATH
       - DATA_URL (str): URL of the dataset to be downloaded from the internet
       - WORKING_DIRECTORY_PATH (str, optional): The working directory in which to 
    store the data fetched. Defaults to a temp directory.
    
        Example usage:
            python -m pydicer.cli.run input --type web 
https://zenodo.org/record/5276878/files/HNSCC.zip cli_test"""

    return help_mesg
