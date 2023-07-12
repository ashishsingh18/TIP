import sys, os
import glob
import reorient
from datetime import datetime
import SimpleITK as sitk

# For local testng
#os.environ["WORKFLOW_DIR"] = "/data"
#os.environ["BATCH_NAME"] = "batch"
#os.environ["OPERATOR_OUT_DIR"] = "output"
#os.environ["OPERATOR_IN_DIR_IMG"] = "None"
#os.environ["OPERATOR_IN_DIR_REF"] = "None"
#os.environ["OPERATOR_IN_DIR_CHOICE"] = "None"
#os.environ["OPERATOR_IN_DIR_EXTENSIONOUT"] = "None"
#os.environ["OPERATOR_IN_DIR_EXTENSIONIMG"] = "None"
#os.environ["OPERATOR_IN_DIR_EXTENSIONREF"] = "None"

# From the template
batch_folders = sorted([f for f in glob.glob(os.path.join('/', os.environ['WORKFLOW_DIR'], os.environ['BATCH_NAME'], '*'))])

for batch_element_dir in batch_folders:
    img = []
    ref = []
    reorient_to = "None"
    extension = ".nii.gz"

    print(f'Checking for nrrd/json files')

    if "None" not in os.environ["OPERATOR_IN_DIR_IMG"]:
        print("img folder provided")
        img_input_dir = os.path.join(batch_element_dir, os.environ['OPERATOR_IN_DIR_IMG'])
        img = sorted(glob.glob(os.path.join(img_input_dir, "*{}*".format(os.environ['OPERATOR_IN_DIR_EXTENSIONIMG'])), recursive=True))

    if "None" not in os.environ["OPERATOR_IN_DIR_REF"]:
        print("ref folder provided")
        ref_input_dir = os.path.join(batch_element_dir, os.environ['OPERATOR_IN_DIR_REF'])
        ref = sorted(glob.glob(os.path.join(ref_input_dir, "*{}*".format(os.environ['OPERATOR_IN_DIR_EXTENSIONREF'])), recursive=True))

    # Only None when you want to match initial image
    if "None" not in os.environ["OPERATOR_IN_DIR_CHOICE"]:
        print("orientation choice provided")
        reorient_to = os.environ["OPERATOR_IN_DIR_CHOICE"]

    if "None" not in os.environ["OPERATOR_EXTENSION"]:
        print("extension choice provided")
        extension = os.environ["OPERATOR_EXTENSION"]

    element_output_dir = os.path.join(batch_element_dir, os.environ['OPERATOR_OUT_DIR'])
        
    # Throws error if ALL components are missing - may want to change in future
    if len(img) == 0 and len(ref) == 0 and reorient_to == "None":
        print("No file(s) found!")
        exit(1)
    else:
        if not os.path.exists(element_output_dir):
            os.makedirs(element_output_dir)

        # Declare UID of subject and where we want to save nifti
        UID = os.path.basename(batch_element_dir)
        outpath = os.path.join(element_output_dir, "{}{}".format(UID,extension))

        # Executing the code
        reorient._main(img,ref,reorient_to,outpath)