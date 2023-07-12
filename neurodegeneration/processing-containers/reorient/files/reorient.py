import SimpleITK as sitk

#read/write utils
def writeimage(image, output_file_path):
    writer = sitk.ImageFileWriter()
    writer.SetFileName ( output_file_path )
    writer.Execute ( image )

def readimage(input_file_path):
    reader = sitk.ImageFileReader()
    reader.SetFileName ( input_file_path )
    image = reader.Execute()
    return image

def reorient_image(image_path, orientation):
    #read input image
    image = readimage(str(image_path))

    #reorient image to input orientation
    reoriented = sitk.DICOMOrient(image, orientation)

    print(reoriented.GetSize())
    
    return reoriented

def reorient_image_to_reference_image(image_path,reference_image_path):
    #read reference image
    refimage = readimage(str(reference_image_path))

    #get direction cosine of reference image
    dc_ref = refimage.GetDirection()

    #instantiate DicomOrientImageFilter & get reference image orientation as string
    orienter = sitk.DICOMOrientImageFilter()
    orientation_ref = orienter.GetOrientationFromDirectionCosines(dc_ref)

    #reorient image to reference image orientation
    reoriented = reorient_image(image_path,orientation_ref)

    print(reoriented.GetSize())

    return reoriented

def _main(img,ref,choice,outpath):
    if choice == "None":
        writeimage(reorient_image_to_reference_image(img[0],ref[0]),outpath)
    else:
        writeimage(reorient_image(img[0],choice),outpath)
