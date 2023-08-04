#!/usr/bin/python3
import tensorflow as tf
from DeepMRSeg import deepmrseg_test, utils
import sys, os, glob
import boto3
import shutil

#for testing
# os.environ["AWS_BATCH_JOB_ID"] = "A" #auto populated
# os.environ["JOB_INPUT_BUCKET"] = "to-kaapana"
# os.environ["JOB_INPUT_KEY"] = "F2.nii.gz"
# os.environ["JOB_REQUESTED_BY_USER"] = "kaapana"
# os.environ['AWS_BATCH_JQ_NAME'] = "C" #auto populated
# os.environ['AWS_BATCH_CE_NAME'] = "D" #auto populated
# os.environ["AWS_SHARED_CREDENTIALS_FILE"] = "/sharedFolder/credentials"

def get_seriesuid(bucket_name,object_key):
  s3 = boto3.client('s3')
  object_metadata = s3.head_object(Bucket=bucket_name, Key=object_key)['Metadata']
  seriesuid = object_metadata['seriesuid']
  print("series uid = ", seriesuid)
  return seriesuid
  

def runProcessing():
    print("Starting deepmrseg based wmls")
    os.makedirs(output_dir, exist_ok=True)
    out_img_nii = output_dir + "output_image.nii.gz"

    #construct cmd to run from command line
    cmd = "deepmrseg_test" + " --mdlDir " + lps_model_path[0] + \
        " --mdlDir " + psl_model_path[0] + \
        " --mdlDir " + slp_model_path[0] + \
        " --inImg " + downloaded_file + \
        " --outImg " + out_img_nii + \
        " --batch " + str(batch_size)

    print("running cmd: ", cmd)
    result = os.system(cmd)

    if result > 0:
        print("wmls exited abnormally. Job failed.")
        sys.exit(1)
    print("wmls finished.")

# AWS Batch Vars
print(f"Batch Job ID: { os.environ['AWS_BATCH_JOB_ID'] }")
print(f"Batch Job Queue: { os.environ['AWS_BATCH_JQ_NAME'] }")
print(f"Batch Job Compute Environment: { os.environ['AWS_BATCH_CE_NAME'] }")
# print(f"Credential file path: { os.environ['AWS_SHARED_CREDENTIALS_FILE'] }")

#setup model dirs
model_dir = "/models/WMLS"
print("model dir: ", model_dir)

batch_size = 4
print('batch_size: ', batch_size)

lps_model_path = [os.path.join(model_dir,"LPS")]
psl_model_path = [os.path.join(model_dir,"PSL")]
slp_model_path = [os.path.join(model_dir,"SLP")]
print("lps model path: ", lps_model_path)
print("psl model path: ", psl_model_path)
print("slp model path: ", slp_model_path)

# CUDA/GPU status
# Check if CUDA/GPU is available
cuda_avail = tf.test.is_gpu_available()

# Get the current GPU device (if available)
current_device = tf.test.gpu_device_name() if cuda_avail else "No GPU available"

# Get a list of available GPU devices
devices = tf.config.list_physical_devices('GPU') if cuda_avail else []

# Get the names of available GPU devices
device_names = [d.name for d in devices] if cuda_avail else []

print("CUDA/GPU status:", cuda_avail)
print("Current GPU device:", current_device)
print("Available GPU devices:", device_names)


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

downloaded_file = "/input/" + input_key

uid = get_seriesuid(input_bucket,input_key)
# Download input file(s) from S3 and place in input
# Placeholder while we figure out more complex batch-subject management
try:
    with open(downloaded_file, 'wb') as f:
        s3.download_fileobj(input_bucket, input_key, f)
    #response = s3.get_object(Bucket=input_bucket, Key=input_key)
    #print("CONTENT TYPE: " + response['ContentType'])
except Exception as e:
    print(e)
    print('Error getting object {} from bucket {}. Make sure they exist and your bucket is in the same region as this function.'.format(input_key, input_bucket))
    raise e

# requestedByUser = os.environ["JOB_REQUESTED_BY_USER"]

#output_dir = os.path.join("/output/", requestedByUser)
output_dir = "/output/"

runProcessing()
# output_archive = os.path.join("/", requestedByUser, job_id)

# shutil.make_archive(output_archive, 'zip', output_dir)
# print(f"Made archive at {output_archive}.zip")
# print(os.listdir("/"))
print(os.listdir(output_dir))

print("Sending to S3...")

out_bucket = "to-kaapana"
#object_name = os.path.join("/private/", requestedByUser, job_id +".zip").lstrip('/')
basename = os.path.basename(input_key).split('.nii.gz',1)[0]
#object_name = "Subject1_0000.nii.gz"
object_name = basename + '_0000.nii.gz'
#file_to_upload = os.path.join("/output/relabeled_out",object_name)
file_to_upload = glob.glob(output_dir + '/*.nii.gz')

s3_upload = boto3.client('s3')
print("Uploading file: ", file_to_upload)
try:
    response = s3_upload.upload_file(
        file_to_upload[0], out_bucket,input_key,
        ExtraArgs={'Metadata': {'seriesuid': uid}}
    )
except Exception as e:
    print(e)
    print('Error uploading object {} from bucket {}. Make sure they exist and your bucket is in the same region as this function.'.format(input_key, out_bucket))
    raise e
print("Object should be available in bucket " + out_bucket + " with object key " + input_key)
print("Done.")

