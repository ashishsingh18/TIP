import os
from os import getenv
from os.path import join, exists, dirname, basename
from glob import glob
from pathlib import Path
import shutil
import SimpleITK as sitk
import boto3
from botocore.exceptions import ClientError

# For multiprocessing -> usually you should scale via multiple containers!
from multiprocessing.pool import ThreadPool

# For shell-execution
from subprocess import PIPE, run

## For local testng
# os.environ["WORKFLOW_DIR"] = "/sharedFolder/nnu1-230609055148242424" #"<your data directory>"
# os.environ["BATCH_NAME"] = "batch"
# os.environ["OPERATOR_IN_DIR"] = "T1_to_nii"
# os.environ["OPERATOR_OUT_DIR"] = "output"
# os.environ["AWS_CREDENTIAL_FILE_PATH"] = "/sharedFolder/credentials"
# os.environ["AWS_CONFIG_FILE_PATH"] = str(None)
# os.environ["AWS_ACCESS_KEY"] = str(None)
# os.environ["AWS_SECRET_KEY"] = str(None)
# os.environ["S3_BUCKET_NAME"] = "from-kaapana"
# os.environ['S3_OBJECT_NAME']= "subject21_0000_0000.nii.gz"
# os.environ["S3_ACTION"] = 'put'

execution_timeout = 300

# Counter to check if smth has been processed
processed_count = 0

aws_credential_file_path=os.environ["AWS_CREDENTIAL_FILE_PATH"]
aws_config_file_path=os.environ["AWS_CONFIG_FILE_PATH"]
aws_access_key=os.environ["AWS_ACCESS_KEY"]
aws_secret_key=os.environ["AWS_SECRET_KEY"]
s3_bucket_name=os.environ["S3_BUCKET_NAME"]
s3_object_name = os.environ['S3_OBJECT_NAME']
s3_action=os.environ["S3_ACTION"]

# set aws specific env variables if specified by user
if(aws_access_key != 'None'):
    os.environ['AWS_ACCESS_KEY_ID']=aws_access_key
    print("os.environ['AWS_ACCESS_KEY_ID']: ", os.environ['AWS_ACCESS_KEY_ID'])
if(aws_secret_key != 'None'):
    os.environ['AWS_SECRET_ACCESS_KEY']=aws_secret_key
    print("os.environ['AWS_SECRET_ACCESS_KEY']: ", os.environ['AWS_SECRET_ACCESS_KEY'])
if(aws_credential_file_path != 'None'):
    os.environ['AWS_SHARED_CREDENTIALS_FILE']=aws_credential_file_path
    print("os.environ['AWS_SHARED_CREDENTIALS_FILE']: ", os.environ['AWS_SHARED_CREDENTIALS_FILE'])
if(aws_config_file_path != 'None'):
    os.environ['AWS_CONFIG_FILE']=aws_config_file_path
    print("os.environ['AWS_CONFIG_FILE']: ", os.environ['AWS_CONFIG_FILE'])

#check valid action
if s3_action not in ['get', 'remove', 'put', 'list']:
    raise AssertionError('action must be get, remove, put or list')

if not os.path.exists('/tempIn'):
   os.makedirs('/tempIn')

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

def write_image(output, output_file_path):
    writer = sitk.ImageFileWriter()
    writer.SetFileName ( str(output_file_path) )
    writer.Execute ( output )

def read_image(input_file_path):
    reader = sitk.ImageFileReader()
    reader.SetFileName ( str(input_file_path) )
    image = reader.Execute()
    return image

#list files in bucket
def list_files(bucket_name):
    s3_client = boto3.client('s3')

    objects = s3_client.list_objects_v2(Bucket=bucket_name)

    for obj in objects['Contents']:
        print(obj['Key'])

    return objects

#upload file
def upload_file(file_name, bucket, object_name=None):
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = os.path.basename(file_name)

    # Upload the file
    s3_client = boto3.client('s3')
    try:
        response = s3_client.upload_file(file_name, bucket, object_name)
    except ClientError as e:
        print(e)
        return False
    return True

def download_file(file_name,bucket_name, object_name=None):
    s3 = boto3.client('s3')
    try:
        response = s3.download_file(bucket_name, object_name, file_name)
    except ClientError as e:
        print(e)
        return False
    return True

def remove_object(bucket_name, object_name):
    s3_client = boto3.client('s3')
    try:
        response = s3_client.delete_object(Bucket=bucket_name,Key=object_name)
    except ClientError as e:
        print(e)

def remove_all_objects(bucket_name):
    response = list_files(bucket_name)
    for object in response['Contents']:
        print('Deleting', object['Key'])
        remove_object(bucket_name, object['Key'])
    
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

    # check if input dir present
    if not exists(element_input_dir):
        print("#")
        print(f"# Input-dir: {element_input_dir} does not exists!")
        print("# -> skipping")
        print("#")
        continue

    # creating output dir
    Path(element_output_dir).mkdir(parents=True, exist_ok=True)

    # creating output dir
    input_files = glob(join(element_input_dir, input_file_extension), recursive=True)
    print(f"# Found {len(input_files)} input-files!")

    # Single process:
    # Loop for every input-file found with extension 'input_file_extension'
    for input_file in input_files:

        print(f'Applying action "{s3_action}" to files {input_file} in S3 bucket "{s3_bucket_name}"')
        if(s3_action == 'list'):
            list_files(s3_bucket_name)
            print("list files done")
        
        if(s3_action == 'put'):#upload file to S3
            upload_file(input_file,s3_bucket_name,s3_object_name)
            print('upload to S3 bucket done.')

        if(s3_action == 'get'):
            output_file_path = os.path.join(element_output_dir, "{}.nii.gz".format(os.path.basename(batch_element_dir)))
            download_file(output_file_path,s3_bucket_name,s3_object_name)
            print("download from S3 bucket done.")

        if(s3_action == 'remove'):
            if((s3_object_name == 'None') or (s3_object_name == '*.*')): #remove all objects from bucket
                remove_all_objects(s3_bucket_name)
            else:#remove specified object from bucket
                remove_object(s3_bucket_name,s3_object_name)
            print("delete from S3 bucket done.")

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

