import sys, os
import glob
import generateBrainVisual
import pickle
from datetime import datetime

# For local testng
#os.environ["WORKFLOW_DIR"] = "/data"
#os.environ["BATCH_NAME"] = "batch"
#os.environ["OPERATOR_OUT_DIR"] = "output"
#os.environ["OPERATOR_IN_DIR_MUSE"] = "deepmrseg"
#os.environ["OPERATOR_IN_DIR_QUANT"] = ''
#os.environ["OPERATOR_IN_DIR_SPARE"] = ''

def write_image(img, output_file_path):
    writer = sitk.ImageFileWriter()
    writer.SetFileName ( output_file_path )
    writer.Execute ( img )

def readimage(input_file_path):
    reader = sitk.ImageFileReader()
    reader.SetFileName ( input_file_path )
    image = reader.Execute()
    return image

# From the template
batch_folders = sorted([f for f in glob.glob(os.path.join('/', os.environ['WORKFLOW_DIR'], os.environ['BATCH_NAME'], '*'))])

for batch_element_dir in batch_folders:

    print(f'Checking for pkl files')
    roi = []

    roi_input_dir = os.path.join(batch_element_dir, os.environ['OPERATOR_IN_DIR_MUSE'])
    roi = sorted(glob.glob(os.path.join(roi_input_dir, "*.nii*"), recursive=True))

    in_dir_quant = os.path.join(batch_element_dir, os.environ['OPERATOR_IN_DIR_QUANT'])
    allz_num = sorted(glob.glob(os.path.join(in_dir_quant, "*allz_num.pkl*"), recursive=True))

    # convert to nrrd 
    # store it to temp location
    # path this to 58

    tmp_path = os.makedir('/tempinput/')
    tmp_file_path = os.path.join(tmp_path, "{}.nrrd".format(os.path.basename(batch_element_dir)))
    input = readimage(roi[0])
    write_image(input, tmp_file_path)

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

    generateBrainVisual._main(tmp_file_path, allz_num, pdf_file_path)
