import sys, os
import glob
import relabel
from datetime import datetime
import SimpleITK as sitk

# For local testng
#os.environ["WORKFLOW_DIR"] = "/data"
#os.environ["BATCH_NAME"] = "batch"
#os.environ["OPERATOR_OUT_DIR"] = "output"
#os.environ["OPERATOR_IN_DIR_ROI"] = "None"
#os.environ["OPERATOR_IN_DIR_CSV"] = "None"

# From the template
batch_folders = sorted([f for f in glob.glob(os.path.join('/', os.environ['WORKFLOW_DIR'], os.environ['BATCH_NAME'], '*'))])

for batch_element_dir in batch_folders:
    roi = []
    csv_file = []

    if "None" not in os.environ["OPERATOR_IN_DIR_ROI"]:
        print("roi folder provided")
        roi_input_dir = os.path.join(batch_element_dir, os.environ['OPERATOR_IN_DIR_ROI'])
        roi = sorted(glob.glob(os.path.join(roi_input_dir, "*.nrrd"), recursive=True))

        print(roi)

    if "None" not in os.environ["OPERATOR_IN_DIR_CSV"]:
        print("csv folder provided")
        #csv_input_dir = os.path.join(batch_element_dir, os.environ['OPERATOR_IN_DIR_CSV'])
        csv_file = os.environ['OPERATOR_IN_DIR_CSV']#sorted(glob.glob(os.path.join(csv_input_dir, "*indices.csv*"), recursive=True))

    element_output_dir = os.path.join(batch_element_dir, os.environ['OPERATOR_OUT_DIR'])
        
    # Throws error if ALL components are missing - may want to change in future
    if len(roi) == 0:
        print("No nifti file(s) found!")
        exit(1)
    else:
        if not os.path.exists(element_output_dir):
            os.makedirs(element_output_dir)

        # Declare UID of subject and where we want to save nifti
        UID = os.path.basename(batch_element_dir)
        outpath = os.path.join(element_output_dir, "{}.nii.gz".format(UID))

        # Write NRRD to NII.GZ
        img = sitk.ReadImage(roi[0])
        sitk.WriteImage(img, outpath)

        # Executing the code
        label_from = 'IndexConsecutive'
        label_to = 'IndexMUSE'
        relabel.relabel_roi_img(outpath,csv_file,label_from,label_to,outpath)