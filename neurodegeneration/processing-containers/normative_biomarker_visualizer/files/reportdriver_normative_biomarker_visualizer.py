import sys, os
import glob
import normative_biomarker_visualizer
import pickle
from datetime import datetime

# # For local testng
# os.environ["WORKFLOW_DIR"] = "/sharedFolder/5datasets_F8F12F18M13M5/F8"
# os.environ["BATCH_NAME"] = "batch"
# os.environ["OPERATOR_OUT_DIR"] = "/sharedFolder/5datasets_F8F12F18M13M5/F8/ouput"
# os.environ["OPERATOR_IN_DIR_ROI"] = "extract_muse_result"
# os.environ["OPERATOR_IN_DIR_QUANT"] = 'roi-quantification'
# os.environ["OPERATOR_IN_DIR_SPARE"] = 'spare-calculation'

# From the template
batch_folders = sorted([f for f in glob.glob(os.path.join('/', os.environ['WORKFLOW_DIR'], os.environ['BATCH_NAME'], '*'))])

for batch_element_dir in batch_folders:

    print(f'Checking for pkl files')

    in_dir_quant = os.path.join(batch_element_dir, os.environ['OPERATOR_IN_DIR_QUANT'])
    in_dir_spare = os.path.join(batch_element_dir, os.environ['OPERATOR_IN_DIR_SPARE'])
    dfSub = sorted(glob.glob(os.path.join(in_dir_quant, "*dfSub.pkl*"), recursive=True))
    dfRef = sorted(glob.glob(os.path.join(in_dir_quant, "*dfRef.pkl*"), recursive=True))
    WMLSref = sorted(glob.glob(os.path.join(in_dir_quant, "*WMLSref.pkl*"), recursive=True))
    allz_num = sorted(glob.glob(os.path.join(in_dir_quant, "*allz_num.pkl*"), recursive=True))
    allz = sorted(glob.glob(os.path.join(in_dir_quant, "*allz.pkl*"), recursive=True))
    all_MuseROIs_name = sorted(glob.glob(os.path.join(in_dir_quant, "*all_MuseROIs_name.pkl*"), recursive=True))
    spareAD = sorted(glob.glob(os.path.join(in_dir_spare, "*spareAD.pkl*"), recursive=True))
    spareBA = sorted(glob.glob(os.path.join(in_dir_spare, "*spareBA.pkl*"), recursive=True))

    element_output_dir = os.path.join(batch_element_dir, os.environ['OPERATOR_OUT_DIR'])
    if not os.path.exists(element_output_dir):
        os.makedirs(element_output_dir)
    
    if len(dfSub) == 0 and len(all_MuseROIs_num) == 0 and len(dfRef) == 0 and len(WMLSref) == 0 and len(allz_num) == 0 and len(allz) == 0 and len(all_MuseROIs_name) == 0 and len(spareAD) == 0 and len(spareBA) == 0:
        print("No file(s) found!")
        exit(1)

    with open(dfSub[0],'rb') as f:
        dfSub = pickle.load(f)
    with open(dfRef[0],'rb') as f:
        dfRef = pickle.load(f)
    with open(WMLSref[0],'rb') as f:
        WMLSref = pickle.load(f)
    with open(allz_num[0],'rb') as f:
        allz_num = pickle.load(f)
    with open(allz[0],'rb') as f:
        allz = pickle.load(f)
    with open(all_MuseROIs_name[0],'rb') as f:
        all_MuseROIs_name = pickle.load(f)
    with open(spareAD[0],'rb') as f:
        spareAD = pickle.load(f)
    with open(spareBA[0],'rb') as f:
        spareBA = pickle.load(f)

    out_path=os.path.join(batch_element_dir, os.environ['OPERATOR_OUT_DIR'])
    pdf_file_path = os.path.join(out_path, "{}.pdf".format(os.path.basename(batch_element_dir)))

    normative_biomarker_visualizer._main(dfSub, dfRef, WMLSref, allz_num, allz, all_MuseROIs_name, spareAD, spareBA, pdf_file_path)