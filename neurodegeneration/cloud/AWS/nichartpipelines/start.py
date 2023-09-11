#!/usr/bin/python3
import torch
import sys
import os
import boto3
import shutil
import time

#for testing
# os.environ["AWS_BATCH_JOB_ID"] = "A" #auto populated
# os.environ["JOB_INPUT_BUCKET"] = "data-coming-from-kaapana-develop"
# os.environ["JOB_INPUT_KEY"] = "Subject1_0000_0000.nii.gz"
# os.environ["JOB_REQUESTED_BY_USER"] = "kaapana"
# os.environ['AWS_BATCH_JQ_NAME'] = "C" #auto populated
# os.environ['AWS_BATCH_CE_NAME'] = "D" #auto populated
# os.environ["AWS_SHARED_CREDENTIALS_FILE"] = "/sharedFolder/v2/credentials"

def get_metadata(bucket_name,object_key):
  s3 = boto3.client('s3')
  object_metadata = s3.head_object(Bucket=bucket_name, Key=object_key)['Metadata']
  seriesuid = object_metadata['seriesuid']
  series_description = object_metadata['seriesdescription']
  print("series uid = ", seriesuid)
  print(" series description = ", series_description)
  return seriesuid,series_description
  

def runProcessing():
    print("Starting nnunet pipeline[dlicv->apply mask->muse->relabel]")
    os.makedirs(output_dir, exist_ok=True)

    result = os.system("niCHARTPipelines -i /input/ -o /output/ " +
        "-p structural " +
        "--derived_ROI_mappings_file /mappings/MUSE_mapping_derived_rois.csv " +
        "--MUSE_ROI_mappings_file /mappings/MUSE_mapping_consecutive_indices.csv " +
        "--model_folder /models " +
        "--all_in_gpu True --mode fastest --disable_tta")

    if result > 0:
        print("nnunet pipeline exited abnormally. Job failed.")
        sys.exit(1)
    print("nnunet pipeline succeeded.")

st = time.time()
# AWS Batch Vars
print(f"Batch Job ID: { os.environ['AWS_BATCH_JOB_ID'] }")
print(f"Batch Job Queue: { os.environ['AWS_BATCH_JQ_NAME'] }")
print(f"Batch Job Compute Environment: { os.environ['AWS_BATCH_CE_NAME'] }")
# print(f"Credential file path: { os.environ['AWS_SHARED_CREDENTIALS_FILE'] }")

# CUDA/GPU status
cuda_avail = torch.cuda.is_available()
current_device = torch.cuda.current_device()
devices = [d for d in range(torch.cuda.device_count())]
device_names = [torch.cuda.get_device_name(d) for d in devices]
print(f"CUDA Available?: {cuda_avail}")
print(f"Devices: {devices}")
print(f"Device Names: {device_names}")
print(f"Current Device: {current_device}")


job_id = os.environ["AWS_BATCH_JOB_ID"]
input_bucket = os.environ["JOB_INPUT_BUCKET"]
input_key = os.environ["JOB_INPUT_KEY"]
# Try printing image info
if not job_id:
    raise RuntimeError("Not running in an AWS Batch environment or Job ID not found!")
if not input_bucket or not input_key:
    raise RuntimeError("Couldn't find input from AWS. Did you run this from the jobprocessor lambda?")

#boto3 code
s3 = boto3.client('s3')

os.makedirs("/input/", exist_ok=True)
os.makedirs("/output/", exist_ok=True)

#TODO: make this generic
folder_name, file_name = os.path.split(input_key)
print("folder name: ", folder_name, "file name: ", file_name)

downloaded_file = "/input/" + file_name

seriesuid,series_description = get_metadata(input_bucket,input_key)
# Download input file(s) from S3 and place in input
# Placeholder while we figure out more complex batch-subject management
try:
    with open(downloaded_file, 'wb') as f:
        s3.download_fileobj(input_bucket, input_key, f)
except Exception as e:
    print(e)
    print('Error getting object {} from bucket {}. Make sure they exist and your bucket is in the same region as this function.'.format(input_key, input_bucket))
    raise e

output_dir = "/output/"

runProcessing()

print("Sending to S3...")

#output bucket name
out_bucket = "to-kaapana"

#create a zip archive containing all output
basename = os.path.basename(input_key).split('_0000_0000.nii.gz',1)[0]
output_archive = os.path.join("/", basename)
shutil.make_archive(output_archive, 'zip', output_dir)
print(f"Made archive at {output_archive}.zip")

#create object name and file to upload
object_name = input_key.split('_0000_0000.nii.gz',1)[0] + '.zip'
file_to_upload = output_archive + ".zip"
print("file to upload: ", file_to_upload)
print('object name: ', object_name)

#upload to output bucket
s3_upload = boto3.client('s3')
print("Uploading file: ", file_to_upload)
try:
    response = s3_upload.upload_file(
        file_to_upload, out_bucket,object_name,
        ExtraArgs={'Metadata': {'seriesuid': seriesuid,'seriesdescription':series_description}}
    )
except Exception as e:
    print(e)
    print('Error uploading object {} from bucket {}. Make sure they exist and your bucket is in the same region as this function.'.format(input_key, out_bucket))
    raise e
print("Object should be available in bucket " + out_bucket + " with object name " + object_name)
print("Done.")

et = time.time()
elapsed_time = (et-st)/60 #mins
print("Elapsed time in mins: ", elapsed_time)