
import pandas as pd
import SimpleITK as sitk

###----image read/write
def read_image(input_file_path):
    reader = sitk.ImageFileReader()
    reader.SetFileName ( str(input_file_path) )
    image = reader.Execute()
    return image

def write_image(output, output_file_path):
    writer = sitk.ImageFileWriter()
    writer.SetFileName ( str(output_file_path) )
    writer.Execute ( output )

def relabel_roi_img(in_img_file, roi_map_file, label_from, label_to, out_img_file):
    '''Convert labels in input roi image to new labels based on the mapping
    The mapping file should contain numeric indices for the mapping
    between the input roi image (from) and output roi image (to)
    '''
    # read input image
    image = read_image(in_img_file)

    ## Read dictionary with roi index mapping 
    df_dict = pd.read_csv(roi_map_file)
    
    #convert dataframe to list
    v_from = df_dict[label_from].to_list()
    v_to = df_dict[label_to].to_list()

    #construct mapping dictionary from lists
    map_dict = dict(zip(v_from,v_to))

    #change label using sitk
    output = sitk.ChangeLabel(image, changeMap=map_dict)

    #write new image
    write_image(output,out_img_file)