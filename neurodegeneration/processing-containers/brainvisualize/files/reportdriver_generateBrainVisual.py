import sys, os
import glob
import generateBrainVisual
import pickle
from datetime import datetime
import SimpleITK as sitk
# For local testng
#os.environ["WORKFLOW_DIR"] = "/data"
#os.environ["BATCH_NAME"] = "batch"
#os.environ["OPERATOR_OUT_DIR"] = "output"
#os.environ["OPERATOR_IN_DIR_MUSE"] = "deepmrseg"
#os.environ["OPERATOR_IN_DIR_QUANT"] = ''
#os.environ["OPERATOR_IN_DIR_SPARE"] = ''

# From the template
batch_folders = sorted([f for f in glob.glob(os.path.join('/', os.environ['WORKFLOW_DIR'], os.environ['BATCH_NAME'], '*'))])

for batch_element_dir in batch_folders:

    print(f'Checking for pkl files')
    roi = []

    roi_input_dir = os.path.join(batch_element_dir, os.environ['OPERATOR_IN_DIR_MUSE'])
    roi = sorted(glob.glob(os.path.join(roi_input_dir, "*.nii*"), recursive=True))

    in_dir_quant = os.path.join(batch_element_dir, os.environ['OPERATOR_IN_DIR_QUANT'])
    allz_num = sorted(glob.glob(os.path.join(in_dir_quant, "*allz_num.pkl*"), recursive=True))

    element_output_dir = os.path.join(batch_element_dir, os.environ['OPERATOR_OUT_DIR'])
    if not os.path.exists(element_output_dir):
        os.makedirs(element_output_dir)

    
    if len(allz_num) == 0:
        print("No file(s) found!")
        exit(1)

    with open(allz_num[0],'rb') as f:
        allz_num = pickle.load(f)

    out_path=os.path.join(batch_element_dir, os.environ['OPERATOR_OUT_DIR'])
    pdf_file_path = os.path.join(out_path, "{}.pdf".format(os.path.basename(batch_element_dir)))

    generateBrainVisual._main(roi[0], allz_num, pdf_file_path)
