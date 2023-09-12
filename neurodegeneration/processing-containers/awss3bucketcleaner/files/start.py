import os
from os import getenv
from os.path import join, exists, dirname, basename
from glob import glob
from pathlib import Path
import shutil
import uuid
import boto3
from botocore.exceptions import ClientError

## For local testng
# os.environ["WORKFLOW_DIR"] = "/sharedFolder/nnu1-230609055148242424" #"<your data directory>"
# os.environ["BATCH_NAME"] = "batch"
# os.environ["OPERATOR_IN_DIR"] = "T1_to_nii"
# os.environ["OPERATOR_OUT_DIR"] = "output"
#--
# os.environ["AWS_CREDENTIAL_FILE_PATH"] = "/sharedFolder/credentials"
# os.environ["AWS_CONFIG_FILE_PATH"] = str(None)
# os.environ["AWS_ACCESS_KEY"] = str(None)
# os.environ["AWS_SECRET_KEY"] = str(None)
# os.environ["S3_BUCKET_NAME"] = "to-kaapana"
# os.environ["S3_ACTION"] = 'empty'

execution_timeout = 300

aws_credential_file_path=os.environ["AWS_CREDENTIAL_FILE_PATH"]
aws_config_file_path=os.environ["AWS_CONFIG_FILE_PATH"]
aws_access_key=os.environ["AWS_ACCESS_KEY"]
aws_secret_key=os.environ["AWS_SECRET_KEY"]
s3_bucket_name=os.environ["S3_BUCKET_NAME"]
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
if s3_action not in ['empty']:
    raise AssertionError('action must be = empty')

def empty_bucket(bucket_name):
    print("######Entered empty_bucket#######")
    print("bucket name: ", bucket_name)
    s3 = boto3.resource('s3')
    try:
        s3.Bucket(bucket_name).objects.delete()
    except s3.meta.client.exceptions as e:
        print(e)
        print(f'Error emptying - bucket: {bucket_name}.')
    

print(f'Applying action "{s3_action}" to S3 bucket "{s3_bucket_name}"')
if(s3_action == 'empty'):
    empty_bucket(s3_bucket_name)
    print(f'S3 bucket {s3_bucket_name} emptied.')
else:
    print("unsupported s3 action given: ", s3_action)

print("# DONE #")

