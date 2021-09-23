###################### Read Me #######################
# This file is to convert .dicom files into .nifti files for PET imaging data, which can work on all patients.
# Below is an example about how to use it.
# BTW, from pet_to_nifti_convert_new import convert_dicom_to_nifty is an important function for converting PET's nifti files.
#############Coded by Shuchao Pang 13 NOV 2020########

import sys
import os
sys.path.append('../../..')
import SimpleITK as sitk
import pydicom
import glob

#from platipy.dicom.rtstruct_to_nifti.convert import convert_rtstruct
#from pet_to_nifti_convert import convert_dicom
from pet_to_nifti_convert_new import convert_dicom_to_nifty

#dicom_search_path = "D:/Data/TCIA----Head and Neck Cancer/Head-Neck-PET-CT" # Location where Dicom data is stored
dicom_search_path = "D:/New_Data/HEAD-NECK-RADIOMICS-HN1" # HN-137
data_path = 'D:/New_Data/000NewCodes/niffty-PET-HN-137-new' # Path where converted files should be placed


data_objects = {}
number_case = 0

for root, dirs, files in os.walk(dicom_search_path, topdown=False):
    for name in files:
        f = os.path.join(root, name)
        ds = pydicom.read_file(f, force=True)
        
        if not 'Modality' in ds:
            continue
        
        case = ds.PatientID.split(' ')[0]

        # if not case in data_objects:
        #     data_objects[case] = {}
        
        p = f
        if ds.Modality == 'CT':
            p = os.path.dirname(f)
            
        if ds.Modality == 'PT':
            p = os.path.dirname(f)

        if ds.Modality == 'PT':
            if not case in data_objects:
                data_objects[case] = {}

            do = {
                'seriesInstanceUID': ds.SeriesInstanceUID, 
                'path': p}
            data_objects[case][ds.Modality] = do
            print("%s --> %s"%(case, ds.Modality))
            number_case += 1

        if ds.Modality == 'RTSTRUCT': #and ('PETPET' in f)): # there are many RTSTRUCT files, so I give a selection
            if not case in data_objects:
                data_objects[case] = {}

            do = {
                'seriesInstanceUID': ds.SeriesInstanceUID, 
                'path': p}
            data_objects[case][ds.Modality] = do
            # print("%s --> %s"%(case, ds.Modality))
            # number_case += 1

        break # Only read one file per dir


print('=========there are totally successful %d cases========='% (number_case))


for case in data_objects:
    do = data_objects[case]
    if 'PT' in do:
        print(case)

        case_dir = os.path.join(data_path, case)
        if not os.path.exists(case_dir):
            os.makedirs(case_dir)
        # do = data_objects[case]
       
        # Convert the PT and RT Struct
        pt_path = do['PT']['path']
        rtstruct_path = do['RTSTRUCT']['path']
        #convert_rtstruct(pt_path, rtstruct_path, output_img='PET.nii.gz', output_dir=case_dir) # this is only for CT or MRI
        # convert_dicom(pt_path, case_dir) # seems not correct for some cases with BQML
        files_pt_path = glob.glob(pt_path +'/*.dcm') 
        sitk_writer = sitk.ImageFileWriter()
        sitk_writer.SetImageIO('NiftiImageIO')
        try:
            convert_dicom_to_nifty(files_pt_path, case_dir, modality='PT',sitk_writer=sitk_writer, extension='.nii.gz')

        except MissingWeightException:
            #ct_path = do['CT']['path']
            # files_ct_path = glob.glob(ct_path + '/*.dcm')
            # header_ct = pydicom.dcmread(files_ct_path[0],
            #                                  stop_before_pixels=True)
            rtstruct_path = do['RTSTRUCT']['path']
            header_rt = pydicom.dcmread(rtstruct_path,
                                             stop_before_pixels=True)
            patient_weight = header_rt.PatientWeight
            if patient_weight is None:
                patient_weight = 75.0  # From Martin's code
                warnings.warn(
                    "Cannot find the weight of the patient, hence it "
                    "is approximated to be 75.0 kg")
            convert_dicom_to_nifty(files_pt_path, case_dir, modality='PT',sitk_writer=sitk_writer, patient_weight_from_ct=patient_weight, extension='.nii.gz')
