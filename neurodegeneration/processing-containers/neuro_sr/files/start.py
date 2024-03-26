import os, sys, csv
from os import getenv
from os.path import join, exists, dirname, basename
from glob import glob
from pathlib import Path
import shutil, json
import SimpleITK as sitk
# import boto3
# from botocore.exceptions import ClientError

# For multiprocessing -> usually you should scale via multiple containers!
from multiprocessing.pool import ThreadPool

# For shell-execution
from subprocess import PIPE, run

## For local testng
# os.environ["WORKFLOW_DIR"] = "/sharedFolder/F12" #"<your data directory>"
# os.environ["BATCH_NAME"] = "batch"
# os.environ["OPERATOR_IN_DIR"] = "T1"
# os.environ["OPERATOR_OUT_DIR"] = "output"
# os.environ["NII_SEGMENTATION"] = "extract_dlicv_result" #"wmls-output"
# os.environ["SERIES_DESCRIPTION"] = "Neuro AD Measurements"
# os.environ["INFO_CSV_DIR"] = "result-generator" 
# os.environ["SERIES_NUMBER"] = "905" 
# os.environ["INSTANCE_NUMBER"] = "1"

#windows
# os.environ["WORKFLOW_DIR"] = "D:\\ashish\\work\\projects\\KaapanaStuff\\dockerContainers\\NeuroSRGenerator\\F12" #"<your data directory>"
# os.environ["BATCH_NAME"] = "batch"
# os.environ["OPERATOR_IN_DIR"] = "Flair"
# os.environ["OPERATOR_OUT_DIR"] = "output"
# os.environ["NII_SEGMENTATION"] = "wmls-output" #"extract_dlicv_result" #"wmls-output"
# os.environ["SEG_INFO_JSON"] = "D:\\ashish\\work\\projects\\KaapanaStuff\\dockerContainers\\NeuroSRGenerator\\0.1.0\\files\\seg_info.json"
# os.environ["SR_INFO_JSON"] = "D:\\ashish\\work\\projects\\KaapanaStuff\\dockerContainers\\NeuroSRGenerator\\0.1.0\\files\\sr_info.json"
# os.environ["INFO_CSV_DIR"] = "result-generator" 
# os.environ["SERIES_NUMBER"] = "300" 
# os.environ["INSTANCE_NUMBER"] = "1"

execution_timeout = 300

# Counter to check if smth has been processed
processed_count = 0

nii_seg = os.environ["NII_SEGMENTATION"]
csv_info_dir = os.environ["INFO_CSV_DIR"]
series_num = os.environ["SERIES_NUMBER"]
instance_num = os.environ["INSTANCE_NUMBER"]
seg_info_json = '/kaapana/app/seg_info.json'
sr_info_json = '/kaapana/app/sr_template.json'
series_desc = os.environ["SERIES_DESCRIPTION"]

if not os.path.exists('/tempOut'):
   os.makedirs('/tempOut')

def copy_file(src,dst):
    shutil.copyfile(src, dst)

def remove_files_from_folder(folder):
    folder += '/*.*'
    files = glob(folder, recursive=True)
    print(files)
    for f in files:
        os.remove(f)
        print('removing temp file: ', f)

def create_dicom_seg(input_dicom_folder,nii_seg,output_dicom_seg,seg_info_json):
    print("Starting dicom seg creation")
    cmd = "/dcmqi-1.3.1-linux/bin/itkimage2segimage" + \
    ' --inputDICOMDirectory ' + \
    input_dicom_folder + \
    ' --inputImageList ' + \
    nii_seg + \
    ' --outputDICOM ' + \
    output_dicom_seg + \
    ' --inputMetadata ' + \
    seg_info_json
    print('running cmd: ', cmd)
    result = os.system(cmd)
    if result > 0:
        print("dicom seg creation failed.")
        sys.exit(1)
    print("dicom seg creation done.")
    
def create_dicom_sr(input_dicom_folder,dcm_seg,output_dicom_sr,sr_info_json):
    print("Starting dicom sr creation")

    cmd = "/dcmqi-1.3.1-linux/bin/tid1500writer" + \
    ' --inputImageLibraryDirectory ' + \
    input_dicom_folder + \
    ' --inputCompositeContextDirectory ' + \
    dcm_seg + \
    ' --outputDICOM ' + \
    output_dicom_sr + \
    ' --inputMetadata ' + \
    sr_info_json
    print('running cmd: ', cmd)
    result = os.system(cmd)
    if result > 0:
        print("dicom sr creation failed.")
        sys.exit(1)
    print("dicom sr creation done.")

def get_sr_values_from_csv(csv_path):
    with open(csv_path, 'r') as file:
        csv_reader = csv.DictReader(file)
        count = 0
        for row in csv_reader:
            data = row
        sr_keys = ['SPARE_AD','SPARE_BA',
        'Total White Matter Hyperintensity Volume',
        'Left Hippocampus Volume','Right Hippocampus Volume',
        'Total Brain Volume']
        sr_data_dict = {}
        for i in  sr_keys:
            sr_data_dict[i] = '{:.2f}'.format(float(data[i]))
        data.clear()
        return sr_data_dict

def read_json_file(json_file):
    with open (json_file, "r") as f:
        data = json.load(f)
    return data

def readimage(input_file_path):
    reader = sitk.ImageFileReader()
    reader.SetFileName ( input_file_path )
    reader.ReadImageInformation()
    image = reader.Execute()
    return image

def create_subject_sr_metajson(json_file_in, dicom_dir, sr_data_dict, dcm_seg_file, json_file_out):
    print('starting creation of subject sr metajson')
    f1 = open(json_file_in, 'r')
    f2 = open(json_file_out, 'w')

    filenames = os.listdir(dicom_dir)
    fileString = '",\n  "'.join(map(os.path.basename, filenames))
    # data['imageLibrary'] = filenames

    #get dicom tags
    dcm_file = join(dicom_dir, filenames[0])
    src_img = readimage(dcm_file)
    seg_img = readimage(dcm_seg_file)

    seriesinstanceUID = src_img.GetMetaData("0020|000e")
    sopintanceUID = seg_img.GetMetaData("0008|0018")
    print("seriesUID: ", str(seriesinstanceUID))
    print("sopUID: ", str(sopintanceUID))

    dcm_seg_basename = os.path.basename(dcm_seg_file)

    for line in f1:

        line = line.replace("<SERIES_DESCRIPTION>", series_desc)
        line = line.replace("<SERIES_NUM>", sr_data_dict["seriesNumber"])
        line = line.replace("<INSTANCE_NUM>", sr_data_dict["instanceNumber"])
        line = line.replace("<DCM_SEGMENTATION_FILE>", dcm_seg_basename)
        line = line.replace("<INSTANCE_FILE_LIST>", fileString)
        line = line.replace("<SERIES_UID>", seriesinstanceUID)
        line = line.replace("<SEGMENTATION_UID>", sopintanceUID)
        line = line.replace("<LEFT_HIPPOCAMPUS_VOLUME>", sr_data_dict["Left Hippocampus Volume"])
        line = line.replace("<RIGHT_HIPPOCAMPUS_VOLUME>", sr_data_dict["Right Hippocampus Volume"])
        line = line.replace("<TOTAL_BRAIN_VOLUME>", sr_data_dict["Total Brain Volume"])
        line = line.replace("<TOTAL_WMH_VOLUME>", sr_data_dict["Total White Matter Hyperintensity Volume"])
        line = line.replace("<SPARE_AD>", sr_data_dict["SPARE_AD"])
        line = line.replace("<SPARE_BA>", sr_data_dict["SPARE_BA"])

        f2.write(line)

    f1.close()
    f2.close()

def update_json_for_subject(json_file_template, dicom_dir, sr_data_dict, dcm_seg_file, sub_json_file_out):
    data = read_json_file(json_file_template)

    filenames = os.listdir(dicom_dir)
    data['imageLibrary'] = filenames

    #get dicom tags
    dcm_file = join(dicom_dir, filenames[0])
    src_img = readimage(dcm_file)
    seg_img = readimage(dcm_seg_file)

    seriesinstanceUID = src_img.GetMetaData("0020|000e")
    sopintanceUID = seg_img.GetMetaData("0008|0018")
    print("seriesUID: ", str(seriesinstanceUID))
    print("sopUID: ", str(sopintanceUID))

    dcm_seg_basename = os.path.basename(dcm_seg_file)
    data["compositeContext"] = [dcm_seg_basename]
    data["SeriesNumber"] = sr_data_dict["seriesNumber"]
    data["InstanceNumber"] = sr_data_dict["instanceNumber"]
    data["SeriesDescription"] = sr_data_dict["seriesdescription"]
    data["Measurements"][0]["SourceSeriesForImageSegmentation"] = seriesinstanceUID
    data["Measurements"][0]["segmentationSOPInstanceUID"] = sopintanceUID
    for item in data["Measurements"][0]["measurementItems"]:
        if(item["value"] == "<LEFT_HIPPOCAMPUS_VOLUME>"):
            item["value"] = sr_data_dict["Left Hippocampus Volume"]
        elif(item["value"] == "<RIGHT_HIPPOCAMPUS_VOLUME>"):
            item["value"] = sr_data_dict["Right Hippocampus Volume"]
        elif(item["value"] == "<TOTAL_BRAIN_VOLUME>"):
            item["value"] = sr_data_dict["Total Brain Volume"]
        elif(item["value"] == "<TOTAL_WMH_VOLUME>"):
            item["value"] = sr_data_dict["Total White Matter Hyperintensity Volume"]
        elif(item["value"] == "<SPARE_AD>"):
            item["value"] = sr_data_dict["SPARE_AD"]
        elif(item["value"] == "<SPARE_BA>"):
            item["value"] = sr_data_dict["SPARE_BA"]
    
    # Write the updated data back to the JSON file
    with open(sub_json_file_out, 'w') as json_file2:
        json.dump(data, json_file2, indent=2)

workflow_dir = getenv("WORKFLOW_DIR", "None")
workflow_dir = workflow_dir if workflow_dir.lower() != "none" else None
assert workflow_dir is not None

batch_name = getenv("BATCH_NAME", "None")
batch_name = batch_name if batch_name.lower() != "none" else None
assert batch_name is not None

operator_in_dir = getenv("OPERATOR_IN_DIR", "None")
operator_in_dir = operator_in_dir if operator_in_dir.lower() != "none" else None
assert operator_in_dir is not None

operator_out_dir = getenv("OPERATOR_OUT_DIR", "None")
operator_out_dir = operator_out_dir if operator_out_dir.lower() != "none" else None
assert operator_out_dir is not None

# File-extension to search for in the input-dir
input_file_extension = "*.nii.gz"

# How many processes should be started?
parallel_processes = 1

print("##################################################")
print("#")
print("# Starting operator awsS3DataMgmt:")
print("#")
print(f"# workflow_dir:     {workflow_dir}")
print(f"# batch_name:       {batch_name}")
print(f"# operator_in_dir:  {operator_in_dir}")
print(f"# operator_out_dir: {operator_out_dir}")
print("#")
print("##################################################")
print("#")
print("# Starting processing on BATCH-ELEMENT-level ...")
print("#")
print("##################################################")
print("#")

# Loop for every batch-element (usually series)
batch_folders = sorted([f for f in glob(join("/", workflow_dir, batch_name, "*"))])
for batch_element_dir in batch_folders:
    print("#")
    print(f"# Processing batch-element {batch_element_dir}")
    print("#")
    element_input_dir = join(batch_element_dir, operator_in_dir)
    element_output_dir = join(batch_element_dir, operator_out_dir)
    # dcm_json_file = sorted(glob.glob(os.path.join(dcm_json_input_dir, "*.json*"), recursive=True))
    nii_seg_dir = join(batch_element_dir,nii_seg)

    print('element_input_dir: ', element_input_dir)
    print('element_output_dir: ', element_output_dir)
    # print('nii_seg_file: ', nii_seg_file)
    print('seg_info_json: ', seg_info_json)
    # check if input dir present
    if not exists(element_input_dir):
        print("#")
        print(f"# Input-dir: {element_input_dir} does not exists!")
        print("# -> skipping")
        print("#")
        continue

    # creating output dir
    Path(element_output_dir).mkdir(parents=True, exist_ok=True)

    #read sr info from csv
    print("csv info dir: ", csv_info_dir)
    csv_dir = join(batch_element_dir,csv_info_dir)
    print('csv dir: ', csv_dir)
    csv_info_file = glob(join(csv_dir, "*.csv"), recursive=True)[0]
    print("csv info file: ", csv_info_file)
    sr_data_dict = get_sr_values_from_csv(csv_info_file)
 

    #add series number and instance number to sr_dict
    sr_data_dict["seriesNumber"] = series_num
    sr_data_dict["instanceNumber"] = instance_num
    sr_data_dict["seriesdescription"] = series_desc
    sr_data_dict["seg_context"] = "DCM_SEGMENTATION_FILE"
    #add instance file list placeholder
    sr_data_dict["instance_file_list"] = "INSTANCE_FILE_LIST"
    sr_data_dict["series_uid"] = "SERIES_UID"
    sr_data_dict["seg_uid"] = "SEGMENTATION_UID"

    print(sr_data_dict)
    # dcm_seg_obj = ""
    # update_json_for_subject(sr_info_json,element_input_dir,sr_data_dict,dcm_seg_obj,"updated_sr_info.json")
    # exit()
    # creating output dir
    nii_seg_file = glob(join(nii_seg_dir, input_file_extension), recursive=True)[0]
    print('nii_seg_file: ', nii_seg_file)

    #create dicom seg object
    out_dcm_seg_file = join(element_output_dir,"dicomsegT1.dcm")
    #out_dcm_seg_file = join("/tempOut","dicomsegT1.dcm")
    print("out_dcm_seg_file: ", out_dcm_seg_file)
    create_dicom_seg(element_input_dir,nii_seg_file,out_dcm_seg_file,seg_info_json)

    #create SR info dict
    subject_sr_file = "/kaapana/app/subject_sr_info.json"
    update_json_for_subject(sr_info_json,element_input_dir,sr_data_dict,out_dcm_seg_file,subject_sr_file)
    # create_subject_sr_metajson(sr_info_json,element_input_dir,sr_data_dict,out_dcm_seg_file,"subject_sr_info.json")
    # exit()
    #create dicom sr object
    out_dcm_sr_file = join(element_output_dir,"dicomsrT1.dcm")
    print("out_dcm_sr_file: ", out_dcm_sr_file)
    print('sr_info_json: ', sr_info_json)
    create_dicom_sr(element_input_dir,element_output_dir,out_dcm_sr_file,subject_sr_file)
    #create_dicom_sr(element_input_dir,out_dcm_seg_file,out_dcm_sr_file,"subject_sr_info.json")
    processed_count += 1

print("#")
print("##################################################")
print("#")
print("# BATCH-ELEMENT-level processing done.")
print("#")
print("##################################################")
print("#")

if processed_count == 0:
    print("#")
    print("##################################################")
    print("#")
    print("##################  ERROR  #######################")
    print("#")
    print("# ----> NO FILES HAVE BEEN PROCESSED!")
    print("#")
    print("##################################################")
    print("#")
    exit(1)
else:
    print("#")
    print(f"# ----> {processed_count} FILES HAVE BEEN PROCESSED!")
    print("#")
    print("# DONE #")

