 #!/usr/bin/env python

import SimpleITK as sitk
import sys, os, glob, json, time, csv
from datetime import datetime
import numpy as np

# For local testng
# os.environ["WORKFLOW_DIR"] = "/sharedFolder/F1" #"<your data directory>"
# os.environ["BATCH_NAME"] = "batch"
# os.environ["OPERATOR_IN_DIR"] = "None"
# os.environ["OPERATOR_OUT_DIR"] = "output"
# os.environ["OPERATOR_IN_MASK_DIR"] = "wmls"
# os.environ["OPERATOR_IN_DCM_JSON_DIR"] = "GetFlairMetadata"
# os.environ["OPERATOR_IN_REFERENCE_IMAGE_DIR"] = "Flair_to_nii"
# os.environ["SERIES_DESC"] = "overlay"
# os.environ["MODALITY"] = "MR"
# os.environ["SERIES_NUM"] = "901"
# os.environ["OPACITY"] = "0.5"
# os.environ["COLOR_SCHEME"] = "/sharedFolder/colors/wmls_color_scheme.csv"

user_specified_modality = os.environ["MODALITY"]
user_specified_series_number = os.environ["SERIES_NUM"] #Default Series Number: 901
user_specified_series_description = os.environ["SERIES_DESC"] #Default Series Description: "Segmentation Overlay"
user_specified_opacity = float(os.environ["OPACITY"]) #Default opacity: 0.5
user_specified_color_scheme = os.environ["COLOR_SCHEME"] #Path to color scheme file

def writeimage(image, output_file_path):
    writer = sitk.ImageFileWriter()
    writer.SetFileName ( output_file_path )
    writer.Execute ( image )

def readimage(input_file_path):
    reader = sitk.ImageFileReader()
    reader.SetFileName ( input_file_path )
    image = reader.Execute()
    return image

def hex_to_rgb(hex: str):
    hex = hex[1:]
    assert len(hex) == 6
    return [int(hex[i:i + 2], 16) for i in (0, 2, 4)]

def read_color_scheme(csv_path):
    with open(csv_path) as cmap:
        reader = csv.reader(cmap, delimiter=',')

        # Read mapping csv to dictionary
        cdict = {}
        for row in reader:
            #second item is roi number, 4th item is color in hexadecimal
            color = row[3]
            #dict of roi label to color
            key = int(row[1])
            print('key: ', key, ' color: ', color)
            cdict[key] = color
    return cdict

def do_overlay(image, segmentation, color_dict,opacity=0.5):
    nda_mask = sitk.GetArrayFromImage(segmentation)
    nda_img = sitk.GetArrayFromImage(image)
    new_img = np.copy(nda_img)
    new_img2 = np.zeros([new_img.shape[0],new_img.shape[1],new_img.shape[2],3])
    new_img2[:,:,:,0] = new_img
    new_img2[:,:,:,1] = new_img
    new_img2[:,:,:,2] = new_img

    for k in color_dict:
        new_img2[nda_mask == k] += opacity * np.array(hex_to_rgb(color_dict[k]))

    rgb_img = sitk.GetImageFromArray(new_img2)
    rgb_img.CopyInformation(segmentation)

    rgb_img_rescaled = sitk.RescaleIntensity(rgb_img)
    result_img = sitk.Cast(rgb_img_rescaled,sitk.sitkVectorUInt8)
    return result_img

def get_dicom_tags_from_json(dcm_json_file):
    with open(dcm_json_file, 'r') as json_file:
        data = json.load(json_file)

        print("reading dicom metadata from json")
        tags_to_copy = []

        # identify relevant tags from the original meta-data dictionary of input image
        ####patient specific tags########
        patient_id = next((value for key, value in data.items() if 'PatientID' in key),None)
        print("patient id: ", patient_id)
        if(patient_id != None):
            tags_to_copy.append(("0010|0020", patient_id))# Patient ID

        patient_name = next((value for key, value in data.items() if 'PatientName' in key),None)
        print("patient name: ", patient_name)
        if(patient_name != None):
            tags_to_copy.append(("0010|0010", patient_name))# Patient Name

        patient_sex = next((value for key, value in data.items() if 'PatientSex' in key),None)
        print("patient sex: ", patient_sex)
        if(patient_sex != None):
            tags_to_copy.append(("0010|0040", patient_sex))# Patient Sex

        patient_age = [value for key, value in data.items() if 'PatientAge' in key][0]
        print("patient age: ", patient_age)
        if(patient_age != None):
            tags_to_copy.append(("0010|1010", patient_age))# Patient age

        patient_size = next((value for key, value in data.items() if 'PatientSize' in key),None)
        print("patient size: ", patient_size)
        if(patient_size != None):
            tags_to_copy.append(("0010|1020", patient_size))# Patient size

        patient_wt = next((value for key, value in data.items() if 'PatientWeight' in key),None)
        print("patient wt: ", patient_wt)
        if(patient_wt != None):
            tags_to_copy.append(("0010|1030", patient_wt))# Patient wt

        #####study specific tags#####
        study_uid = next((value for key, value in data.items() if 'StudyInstanceUID' in key),None)
        print("study uid ", study_uid)
        if(study_uid != None):
            tags_to_copy.append(("0020|000D", study_uid))# Study Instance UID, for machine consumption

        study_id = next((value for key, value in data.items() if 'StudyID' in key),None)
        print("study id ", study_id)
        if(study_id != None):
            tags_to_copy.append(("0020|0010", study_id))# Study ID, for human consumption

        study_date = next((value for key, value in data.items() if 'StudyDate' in key),None)
        print("study date ", study_date)
        if(study_date != None):
            tags_to_copy.append(("0008|0020", study_date))# Study Date

        study_time = next((value for key, value in data.items() if 'StudyTime' in key),None)
        print("study time ", study_time)
        if(study_time != None):
            tags_to_copy.append(("0008|0030", study_time))# Study Time

        #####other tags####
        #use modality specified by user(think multi-modality pipeline) otherwise use the one from reference image
        if(user_specified_modality == "None"):
            modality = next((value for key, value in data.items() if 'Modality' in key),None)
        else:
            modality = user_specified_modality
        print("modality ", modality)
        if(study_time != None):
            tags_to_copy.append(("0008|0060", modality))  # Modality

        #accession number
        possible_accession_number_values = [value for key, value in data.items() if 'AccessionNumber' in key]
        if len(possible_accession_number_values) == 0:
            attribute_sequence = [value for key, value in data.items() if 'RequestAttributesSequence_object_object' in key][0]
            possible_accession_number_values = [value for key, value in attribute_sequence.items() if 'AccessionNumber' in key]
            if len(possible_accession_number_values) == 0:
                print("accession number not found")
                accession_number_found = False
            else:
                accession_number_found = True
                accession_number = possible_accession_number_values[0]
        else:
            accession_number_found = True
            accession_number = possible_accession_number_values[0]

        # print('accession_number: ', accession_number)

        if(accession_number_found):
            tags_to_copy.append(("0008|0050", accession_number)) #AccessionNumber

    return tags_to_copy

def write_dicom_slices(outdir, tags_to_write, new_img, i):
    image_slice = new_img[:,:,i]

    # Tags shared by the series.
    list(map(lambda tag_value: image_slice.SetMetaData(tag_value[0], str(tag_value[1])), tags_to_write))

    # Slice specific tags.
    image_slice.SetMetaData("0008|0012", time.strftime("%Y%m%d")) # Instance Creation Date
    image_slice.SetMetaData("0008|0013", time.strftime("%H%M%S")) # Instance Creation Time

    #Setting the type to CT preserves the slice location.
    #image_slice.SetMetaData("0008|0060", "MR")  # set the type to CT so the thickness is carried over

    # (0020, 0032) image position patient determines the 3D spacing between slices.
    image_slice.SetMetaData("0020|0032", '\\'.join(map(str,new_img.TransformIndexToPhysicalPoint((0,0,i))))) # Image Position (Patient)
    image_slice.SetMetaData("0020|0013", str(i)) # Instance Number
    image_slice.SetMetaData("0020|0011", str(user_specified_series_number)) # Series Number - default 901

    # Write to the output directory and add the extension dcm, to force writing in DICOM format.
    writer.SetFileName(os.path.join(outdir,str(i)+'.dcm'))
    writer.Execute(image_slice)

batch_folders = [f for f in glob.glob(os.path.join('/', os.environ['WORKFLOW_DIR'], os.environ['BATCH_NAME'], '*'))]
print('batch_folders: ',batch_folders)

for batch_element_dir in batch_folders:

    if "None" not in os.environ["OPERATOR_IN_REFERENCE_IMAGE_DIR"]:
        print("Reference image folder provided")
        ref_image_input_dir = os.path.join(batch_element_dir, os.environ['OPERATOR_IN_REFERENCE_IMAGE_DIR'])
        ref_image_file = sorted(glob.glob(os.path.join(ref_image_input_dir, "*.nii.gz"), recursive=True)) or sorted(glob.glob(os.path.join(ref_image_input_dir, "*.nrrd"), recursive=True))
        print(ref_image_file)

    if "None" not in os.environ["OPERATOR_IN_MASK_DIR"]:
        print("mask image folder provided")
        mask_image_input_dir = os.path.join(batch_element_dir, os.environ['OPERATOR_IN_MASK_DIR'])
        mask_image_file = sorted(glob.glob(os.path.join(mask_image_input_dir, "*.nrrd"), recursive=True)) or sorted(glob.glob(os.path.join(mask_image_input_dir, "*.nii.gz"), recursive=True))
        print(mask_image_file)

    if "None" not in os.environ["OPERATOR_IN_DCM_JSON_DIR"]:
        print("Dicom json(metadata) folder provided")
        dcm_json_input_dir = os.path.join(batch_element_dir, os.environ['OPERATOR_IN_DCM_JSON_DIR'])
        dcm_json_file = sorted(glob.glob(os.path.join(dcm_json_input_dir, "*.json*"), recursive=True))
        print(dcm_json_file)

    if len(ref_image_file) == 0 and len(mask_image_file) == 0 and len(dcm_json_file) == 0 and len(user_specified_color_scheme) == 0:
        print("reference image, mask image, dicom json file or color scheme csv file not found!")
        exit(1)
    else:
        print(f"Starting creation of dicom seg overlay for reference image {ref_image_file} with mask {mask_image_file} and json {dcm_json_file}")

        image = readimage(ref_image_file[0])
        print("input reference image read")

        cDict = read_color_scheme(user_specified_color_scheme)
        print('# items with color mapping: ', len(cDict))

        # To visualize the labels image in RGB we need to reduce the intensity range ( 0-255 )
        img_255 = sitk.Cast(sitk.RescaleIntensity(image), sitk.sitkUInt8)

        mask = readimage(mask_image_file[0])
        print("label map read")

        #do the overlay
        #overlaid_img = sitk.LabelOverlay(image=img_255, labelImage=mask, opacity=0.5)
        overlaid_img = do_overlay(img_255,mask,cDict,user_specified_opacity)
        print("Label Overlay Done")

        #dicom creation
        writer = sitk.ImageFileWriter()
        writer.KeepOriginalImageUIDOn()

        #create new dicom tags for our dicom file
        modification_time = time.strftime("%H%M%S")
        modification_date = time.strftime("%Y%m%d")

        direction = overlaid_img.GetDirection()
        series_tag_values = [("0008|0031",modification_time), # Series Time
                          ("0008|0021",modification_date), # Series Date
                          ("0008|0008","DERIVED\\SECONDARY"), # Image Type
                          ("0020|000e", "1.2.826.0.1.3680043.2.1125."+modification_date+".1"+modification_time), # Series Instance UID
                          ("0020|0037", '\\'.join(map(str, (direction[0], direction[3], direction[6],# Image Orientation (Patient)
                                                            direction[1],direction[4],direction[7])))),
                          ("0008|103e", user_specified_series_description)] # Series Description - default "segmentation overlay"

        #dicom tags to write
        tags_to_write = series_tag_values + get_dicom_tags_from_json(dcm_json_file[0])

        element_output_dir = os.path.join(batch_element_dir, os.environ['OPERATOR_OUT_DIR'])
        if not os.path.exists(element_output_dir):
            os.makedirs(element_output_dir)

        #write dicom images
        list(map(lambda i: write_dicom_slices(element_output_dir,tags_to_write, overlaid_img, i), range(overlaid_img.GetDepth())))
        print("dicom rgb overlay written")
