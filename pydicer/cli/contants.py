def get_sub_help_mesg(input_commands, command):
    # pylint: disable=missing-function-docstring

    help_mesg = f"""Subcommand of the following: {input_commands}

        test WORKING_DIRECTORY_PATH

        Runs the command using the default test data. Check pydicer.input.test for more info

            - WORKING_DIRECTORY_PATH: The working directory in which to
                store the data fetched. Defaults to a temp directory.
        
            Example usage:
                python -m pydicer.cli.run input|pipeline --type test cli_test

        pacs WORKING_DIRECTORY_PATH HOST_IP PORT AE_TITLE MODALITY [PATIENT_IDs]

        Runs the command by querying a DIOCM PACS server and storing the data on locally on the
        filesystem

            - WORKING_DIRECTORY_PATH: The working directory in which to
                store the data fetched. Defaults to a temp directory.
            - HOST_IP (optional): The IP address of host name of DICOM PACS. Defaults to 
                'www.dicomserver.co.uk'.
            - PORT (optional): The port to use to communicate on. Defaults to 11112.
            - AE_TITLE (optional): AE Title to provide the DICOM service. Defaults to 
            None.
            - MODALITY (optional): The modality to retrieve DICOMs for. Defaults 
                to 'GM'.
            - PATIENT_IDs (required): a string-list of patient IDs (IDs seperated by spaces)
                to retrieve the DICOMs for.
        
            Example usage:
                python -m pydicer.cli.run input|pipeline --type pacs www.dicomserver.co.uk 11112
                    DCMQUERY cli_test GM PAT004 PAT005
        
        
        web WORKING_DIRECTORY_PATH DATA_URL

        Runs the command by downloading data from a provided URL and storing it locally on the
        filesystem

        - WORKING_DIRECTORY_PATH: The working directory in which to 
                store the data fetched. Defaults to a temp directory.
        - DATA_URL: URL of the dataset to be downloaded from the internet
        
            Example usage:
                python -m pydicer.cli.run input|pipeline --type web 
                https://zenodo.org/record/5276878/files/HNSCC.zip cli_test
        """
    if command == "pipeline":
        help_mesg += """

        filesystem WORKING_DIRECTORY_PATH

        Runs the pipeline using a filesystem working directory which contains DICOM images as input

            - WORKING_DIRECTORY_PATH: The working directory in which to
                store the data fetched. Defaults to a temp directory.
        
            Example usage:
                python -m pydicer.cli.run pipeline --type filesystem cli_test
        

        e2e

        Runs the entire pipeline using the default settings. Check pydicer.pipeline for more info

            Example usage:
                python -m pydicer.cli.run pipeline
                or
                python -m pydicer.cli.run pipeline --type e2e

        """

    return help_mesg
