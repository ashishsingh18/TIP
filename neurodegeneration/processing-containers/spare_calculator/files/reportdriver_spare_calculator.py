import sys, os
import glob
import spare_calculator
import pickle
from datetime import datetime

# For local testng
#os.environ["WORKFLOW_DIR"] = "/data"
#os.environ["BATCH_NAME"] = "batch"
#os.environ["OPERATOR_OUT_DIR"] = "output"
#os.environ["OPERATOR_IN_DIR"]

# From the template
batch_folders = sorted([f for f in glob.glob(os.path.join('/', os.environ['WORKFLOW_DIR'], os.environ['BATCH_NAME'], '*'))])

for batch_element_dir in batch_folders:

    print(f'Checking for pkl files')

    if "None" not in os.environ["OPERATOR_IN_DIR"]:
        in_dir = os.path.join(batch_element_dir, os.environ['OPERATOR_IN_DIR'])
        dfSub = sorted(glob.glob(os.path.join(in_dir, "*dfSub.pkl*"), recursive=True))
        all_MuseROIs_num = sorted(glob.glob(os.path.join(in_dir, "*all_MuseROIs_num.pkl*"), recursive=True))

    element_output_dir = os.path.join(batch_element_dir, os.environ['OPERATOR_OUT_DIR'])
    if not os.path.exists(element_output_dir):
        os.makedirs(element_output_dir)
    
    if len(dfSub) == 0 and len(all_MuseROIs_num) == 0:
        print("No file(s) found!")
        exit(1)

    with open(dfSub[0],'rb') as f:
        dfSub = pickle.load(f)

    with open(all_MuseROIs_num[0],'rb') as f:
        all_MuseROIs_num = pickle.load(f)

    out_path=os.path.join(batch_element_dir, os.environ['OPERATOR_OUT_DIR'])
    pdf_file_path = os.path.join(out_path, "{}.pdf".format(os.path.basename(batch_element_dir)))

    spare_calculator._main(dfSub,all_MuseROIs_num,pdf_file_path)