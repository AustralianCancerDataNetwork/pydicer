import os
import urllib.request, urllib.error, urllib.parse
import urllib.request, urllib.parse, urllib.error
from pydicer.input.base import InputBase
import zipfile
from pathlib import Path

#
# Refer https://wiki.cancerimagingarchive.net/display/Public/REST+API+Usage+Guide for complete
# list of API
#
# Some of the code was adapted from the TCIA Python APi client example:
# https://github.com/TCIA-Community/TCIA-API-SDK/blob/master/tcia-rest-client-python/src/tciaclient.py
#


def run_endpoint(url, query_parameters):
    """
    Execute the TCIA API endpoint

    Args:
        url (str): url of the series to be downloaded and generate its DICOMS
        query_parameters (dict): dictionary of query parameters to be passed into the api
        endpoint request.

    Returns:
        resp (file): the response file in bytes to be written into a zipfile
    """

    query_parameters = dict((k, v) for k, v in query_parameters.items() if v)
    queryString = "?%s" % urllib.parse.urlencode(query_parameters)
    requestUrl = url + queryString
    request = urllib.request.Request(url=requestUrl)
    resp = urllib.request.urlopen(request)
    return resp


class TCIAInput(InputBase):
    def __init__(self, series_instance_uid, working_directory=None):
        """
        Input class that interfaces with the TCIA API

        Args:
            series_instance_uid (str): The series Instance UID to be downloaded from TCIA
            working_directory (str|pathlib.Path, optional): The working directory in which to
            store the downloaded data. Defaults to a temp directory
        """
        super().__init__(working_directory)
        self.prefix_url = "https://services.cancerimagingarchive.net/services/v4/TCIA"
        self.series_instance_uid = series_instance_uid

    def fetch_data(self, output_zip_file="tcia_input_zipfile.zip"):
        """
        Function to download the Series instance UID from TCIA and write locally

        Args:
            output_zip_file (str): name of temporary zipfile
        """

        serviceUrl = self.prefix_url + "/query/getImage"
        query_parameters = {"SeriesInstanceUID": self.series_instance_uid}
        try:
            self.working_directory.mkdir(exist_ok=True, parents=True)
            file = os.path.join(self.working_directory, output_zip_file)
            resp = run_endpoint(serviceUrl, query_parameters)

            # If series instance uid doesn't exist in TCIA, return
            if not resp.chunked:
                print(
                    "Error: The Series Instance UID provided does not exist, please confirm that it"
                    " is correct..."
                )
                return

            # Chunks to be read by, used by TCIA's official python API client example
            CHUNK = 256 * 10240
            # Open zipfile
            with open(file, "wb") as fp:
                # Keep reading bytes response file
                while True:
                    chunk = resp.read(CHUNK)
                    # Stop reading and writing bytes file when there are no more bytes left
                    if not chunk:
                        break
                    # Write chunk to zipfile
                    fp.write(chunk)
            # Unzip temporary zipfile with DICOMs
            with zipfile.ZipFile(file, "r") as zip_ref:
                zip_ref.extractall(self.working_directory)
            # Remove temporary zipfile once the DICOMs are written
            os.remove(file)
        except Exception as e:
            print(e)
