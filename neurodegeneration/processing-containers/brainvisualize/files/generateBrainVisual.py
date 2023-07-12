import os as _os
import cv2
import pylab as pl
import numpy as np
import pandas as pd
import vtk
import pickle
import SimpleITK as sitk
import sys, glob, csv
import matplotlib
import subprocess

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

from createcmap import get_continuous_cmap

#maphemi = pd.read_csv('/Users/vikasbommineni/Desktop/MRIreport/brainvisualize/refs/MUSE_ROI_Dictionary.csv')
maphemi = pd.read_csv('refs/MUSE_ROI_Dictionary.csv')

# def write_image(img, output_file_path):
#     writer = sitk.ImageFileWriter()
#     writer.SetFileName ( output_file_path )
#     writer.Execute ( img )
    
# def readimage(input_file_path):
#     reader = sitk.ImageFileReader()
#     reader.SetFileName ( input_file_path )
#     image = reader.Execute()
#     return image

def getbounds(arr):
	arr = np.mean(arr,axis=2)
	pos = np.nonzero(arr == 0)

	top = pos[0].min()
	bottom = pos[0].max()

	left = pos[1].min()
	right = pos[1].max()

	return top,bottom,left,right

# Crops each orientation image with 10 pixels boundary
def crop_and_write(img,orientation,fname,x_dim,y_dim):
    t,b,l,r = getbounds(img)

    img = img[(t-10):(b+10),(l-10):(r+10)]
    cv2.imwrite(fname+'_' + orientation + '.png',img)

def vtk_show(orientation, fname, renderer, width=400, height=300):
    """
    Takes vtkRenderer instance and returns an IPython Image with the rendering.
    """
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.SetOffScreenRendering(1)
    renderWindow.AddRenderer(renderer)
    renderWindow.SetSize(width, height)
    renderWindow.Render()
     
    windowToImageFilter = vtk.vtkWindowToImageFilter()
    windowToImageFilter.SetInput(renderWindow)
    windowToImageFilter.Update()
    
    writer = vtk.vtkPNGWriter()
    writer.SetFileName(fname+'_' + orientation + '.png')
    #writer.SetWriteToMemory(1)
    writer.SetInputConnection(windowToImageFilter.GetOutputPort())
    writer.Write()
    
    # Rotate image to allow for good display
    img=cv2.imread(fname+'_' + orientation + '.png')
    
    if orientation == 'axialtop':
        imgrot = cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE)
    elif orientation == 'axialbottom':
        imgrot = cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE)
    elif orientation == 'lefthemosphere_lateral':
        imgrot = cv2.rotate(img,cv2.ROTATE_180)
    elif orientation == 'righthemosphere_lateral':
    	imgrot = cv2.rotate(img,cv2.ROTATE_180)

    ### Delete associated file for each of the following screenshot images to clear up space ###
    elif orientation == 'right_medial':
    	_os.remove(fname+'_r.nii.gz')
    	imgrot = cv2.rotate(img,cv2.ROTATE_180)
    elif orientation == 'left_medial':
    	_os.remove(fname+'_l.nii.gz')
    	imgrot = cv2.rotate(img,cv2.ROTATE_180)
    elif orientation == 'bg_t_axial':
    	_os.remove(fname+'_bg_t.nii.gz')
    	imgrot = cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE)

    crop_and_write(imgrot,orientation,fname,width,height)

def readimage(input_file_path):
    reader = sitk.ImageFileReader()
    reader.SetFileName ( input_file_path )
    image = reader.Execute()
    return image

def write_image(img, output_file_path):
    writer = sitk.ImageFileWriter()
    writer.SetFileName ( output_file_path )
    writer.Execute ( img )

def create_relabel_map(muse_mask,roi_zscore_dict):
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(muse_mask)

    merge_label_dict={}

    for i in stats.GetLabels():
        if str(i) not in roi_zscore_dict.keys():
            merge_label_dict[i] = 220
        else:
            merge_label_dict[i] = roi_zscore_dict[str(i)]

    return merge_label_dict

def atrophyvisualization(maskfile, allz, fname):
	# Path to the file
	filenameSegmentation = maskfile

	muse_labelmap=readimage(filenameSegmentation)
	roi_zscore_dict=allz
	relabelMap = create_relabel_map(muse_labelmap,roi_zscore_dict)

	# Scale values less than -0.524 from yellow to red
	rzd = list(roi_zscore_dict.values())
	rzd = [i for i in rzd if (i <= -0.5244)]
	#min_zscore = min(min(rzd),-3)
	# 1st percentile z-score equivalent
	min_zscore = -2.326
	need2relabel = [int(k) for k,v in roi_zscore_dict.items() if float(v) <= -0.524]
	relabelGray = [int(k) for k in relabelMap.keys() if k not in need2relabel]

	nda_all = sitk.GetArrayFromImage(muse_labelmap)
	nda_l = sitk.GetArrayFromImage(muse_labelmap)
	nda_r = sitk.GetArrayFromImage(muse_labelmap)
	nda_bg_t = sitk.GetArrayFromImage(muse_labelmap)

	# TODO: get equitable split
	bg_t = ((nda_bg_t == 59) | (nda_bg_t == 60) | (nda_bg_t == 23) | (nda_bg_t == 30) | (nda_bg_t == 36) | (nda_bg_t == 37) | (nda_bg_t == 55) | (nda_bg_t == 56) | (nda_bg_t == 57) | (nda_bg_t == 58)).astype(int)
	cnt = np.inf
	bg_t_slice = 0
	for y in range(bg_t.shape[1]):
		if abs(np.sum(bg_t[:,:y,:]) - np.sum(bg_t[:,y:,:])) < cnt:
			bg_t_slice = y
			cnt = abs(np.sum(bg_t[:,:y,:]) - np.sum(bg_t[:,y:,:]))

	# Set voxels to 0 above V.O.I
	nda_bg_t[:,:bg_t_slice,:] = 0

	# Get transition point from left to right hemisphere from corpus callosum
	cc = (nda_all == 95).astype(int)

	#### test slices to get the most equitable cc split
	cnt = np.inf
	mid_slice = 0
	for x in range(cc.shape[0]):
		if abs(np.sum(cc[x:,:,:])-np.sum(cc[:x,:,:])) < cnt:
			mid_slice = x
			cnt = abs(np.sum(cc[x:,:,:])-np.sum(cc[:x,:,:]))

	# Set voxels to 0 based on middle slice
	nda_l[mid_slice:,:,:] = 0
	nda_r[:mid_slice,:,:] = 0

	all_post = sitk.GetImageFromArray(nda_all)
	l_post = sitk.GetImageFromArray(nda_l)
	r_post = sitk.GetImageFromArray(nda_r)
	bg_t_post = sitk.GetImageFromArray(nda_bg_t)

	all_post.CopyInformation(muse_labelmap)
	l_post.CopyInformation(muse_labelmap)
	r_post.CopyInformation(muse_labelmap)
	bg_t_post.CopyInformation(muse_labelmap)

	all_filenameSegmentation = fname+'_all.nii.gz'
	l_filenameSegmentation = fname+'_l.nii.gz'
	r_filenameSegmentation = fname+'_r.nii.gz'
	bg_t_filenameSegmentation = fname+'_bg_t.nii.gz'
	
	print(174)

	write_image(all_post,all_filenameSegmentation)
	write_image(l_post,l_filenameSegmentation)
	write_image(r_post,r_filenameSegmentation)
	write_image(bg_t_post,bg_t_filenameSegmentation)

	##################################################

	reader_all = vtk.vtkNIFTIImageReader()
	reader_l = vtk.vtkNIFTIImageReader()
	reader_r = vtk.vtkNIFTIImageReader()
	reader_bg_t = vtk.vtkNIFTIImageReader()

	reader_all.SetFileName(all_filenameSegmentation)
	reader_l.SetFileName(l_filenameSegmentation)
	reader_r.SetFileName(r_filenameSegmentation)
	reader_bg_t.SetFileName(bg_t_filenameSegmentation)

	castFilter_all = vtk.vtkImageCast()
	castFilter_l = vtk.vtkImageCast()
	castFilter_r = vtk.vtkImageCast()
	castFilter_bg_t = vtk.vtkImageCast()

	castFilter_all.SetInputConnection(reader_all.GetOutputPort())
	castFilter_l.SetInputConnection(reader_l.GetOutputPort())
	castFilter_r.SetInputConnection(reader_r.GetOutputPort())
	castFilter_bg_t.SetInputConnection(reader_bg_t.GetOutputPort())

	castFilter_all.SetOutputScalarTypeToUnsignedShort()
	castFilter_l.SetOutputScalarTypeToUnsignedShort()
	castFilter_r.SetOutputScalarTypeToUnsignedShort()
	castFilter_bg_t.SetOutputScalarTypeToUnsignedShort()

	castFilter_all.Update()
	castFilter_l.Update()
	castFilter_r.Update()
	castFilter_bg_t.Update()

	imdataBrainSeg_all = castFilter_all.GetOutputPort()
	imdataBrainSeg_l = castFilter_l.GetOutputPort()
	imdataBrainSeg_r = castFilter_r.GetOutputPort()
	imdataBrainSeg_bg_t = castFilter_bg_t.GetOutputPort()

	idbs_all = castFilter_all.GetOutput()
	idbs_l = castFilter_l.GetOutput()
	idbs_r = castFilter_r.GetOutput()
	idbs_bg_t = castFilter_bg_t.GetOutput()

	# Define color legend #
	funcColor = vtk.vtkColorTransferFunction()

	for idx in relabelMap.keys():
		if idx in need2relabel:
			# TODO: create a fixed z-score based on most extreme of training data
			if -1.036 <= relabelMap[idx] <= -.5244:
				funcColor.AddRGBPoint(idx,255/255,255/255,178/255)
			elif -1.645 <= relabelMap[idx] < -1.036:
				funcColor.AddRGBPoint(idx,254/255,204/255,92/255)
			elif -1.881 <= relabelMap[idx] < -1.645:
				funcColor.AddRGBPoint(idx,253/255,141/255,60/255)
			elif -2.326 <= relabelMap[idx] < -1.881:
				funcColor.AddRGBPoint(idx,240/255,59/255,32/255)
			elif relabelMap[idx] < -2.326:
				funcColor.AddRGBPoint(idx,189/255,0/255,38/255)
		else:
			funcColor.AddRGBPoint(idx,1,1,1)

	funcColor.AddRGBPoint(0,1,1,1)

	# Define opacity scheme #
	funcOpacityScalar = vtk.vtkPiecewiseFunction()
	funcOpacityScalar.AddPoint(0, 0)
	for idx in relabelMap.keys():
		funcOpacityScalar.AddPoint(idx, 1 if idx > 0 else 0.0)

	### WHOLE VOLUME ###
	propVolume_all = vtk.vtkVolumeProperty()
	propVolume_l = vtk.vtkVolumeProperty()
	propVolume_r = vtk.vtkVolumeProperty()
	propVolume_bg_t = vtk.vtkVolumeProperty()

	propVolume_all.SetColor(funcColor)
	propVolume_l.SetColor(funcColor)
	propVolume_r.SetColor(funcColor)
	propVolume_bg_t.SetColor(funcColor)

	propVolume_all.SetScalarOpacity(funcOpacityScalar)
	propVolume_l.SetScalarOpacity(funcOpacityScalar)
	propVolume_r.SetScalarOpacity(funcOpacityScalar)
	propVolume_bg_t.SetScalarOpacity(funcOpacityScalar)

	propVolume_all.SetInterpolationTypeToNearest()
	propVolume_l.SetInterpolationTypeToNearest()
	propVolume_r.SetInterpolationTypeToNearest()
	propVolume_bg_t.SetInterpolationTypeToNearest()

	propVolume_all.ShadeOn()
	propVolume_l.ShadeOn()
	propVolume_r.ShadeOn()
	propVolume_bg_t.ShadeOn()

	#propVolume_all.SetSpecular(0.8)
	#propVolume_l.SetSpecular(0.8)
	#propVolume_r.SetSpecular(0.8)
	#propVolume_bg_t.SetSpecular(0.8)

	propVolume_all.SetDiffuse(0.7)
	propVolume_l.SetDiffuse(0.7)
	propVolume_r.SetDiffuse(0.7)
	propVolume_bg_t.SetDiffuse(0.7)

	propVolume_all.SetAmbient(0.8)
	propVolume_l.SetAmbient(0.8)
	propVolume_r.SetAmbient(0.8)
	propVolume_bg_t.SetAmbient(0.8)

	volumeMapper_all = vtk.vtkFixedPointVolumeRayCastMapper()
	volumeMapper_l = vtk.vtkFixedPointVolumeRayCastMapper()
	volumeMapper_r = vtk.vtkFixedPointVolumeRayCastMapper()
	volumeMapper_bg_t = vtk.vtkFixedPointVolumeRayCastMapper()

	volumeMapper_all.SetInputConnection(imdataBrainSeg_all)
	volumeMapper_l.SetInputConnection(imdataBrainSeg_l)
	volumeMapper_r.SetInputConnection(imdataBrainSeg_r)
	volumeMapper_bg_t.SetInputConnection(imdataBrainSeg_bg_t)

	volume_all = vtk.vtkVolume()
	volume_l = vtk.vtkVolume()
	volume_r = vtk.vtkVolume()
	volume_bg_t = vtk.vtkVolume()

	volume_all.SetMapper(volumeMapper_all)
	volume_l.SetMapper(volumeMapper_l)
	volume_r.SetMapper(volumeMapper_r)
	volume_bg_t.SetMapper(volumeMapper_bg_t)

	volume_all.SetProperty(propVolume_all)
	volume_l.SetProperty(propVolume_l)
	volume_r.SetProperty(propVolume_r)
	volume_bg_t.SetProperty(propVolume_bg_t)

	normal = [0,0,0]
	viewUp = [0,0,0]

	# Set orientation logic #
	for orientation in ['left_medial','bg_t_axial','right_medial','axialtop','axialbottom','lefthemosphere_lateral','righthemosphere_lateral']:
		if (orientation == 'axialtop') | (orientation == 'axialbottom') | (orientation == 'lefthemosphere_lateral') | (orientation == 'righthemosphere_lateral'):
			renderer = vtk.vtkRenderer()
			renderWin = vtk.vtkRenderWindow()
			renderWin.SetOffScreenRendering(1)

			renderWin.AddRenderer(renderer)
			renderInteractor = vtk.vtkRenderWindowInteractor()
			renderInteractor.SetRenderWindow(renderWin)

			renderer.AddVolume(volume_all)
			renderer.SetBackground((255,255,255))

			if orientation == 'axialtop':
				viewUp = [0, 0, 1]
				normal = [0, -1, 0]
			elif orientation == 'axialbottom':
				viewUp = [0, 0, -1]
				normal = [0, 1, 0]
			elif orientation == 'lefthemosphere_lateral':
				viewUp = [0, 1, 0]
				normal = [0, 0, 1]
			elif orientation == 'righthemosphere_lateral':
				viewUp = [0, 1, 0]
				normal = [0,0,-1]

		elif orientation == 'right_medial':
			renderer = vtk.vtkRenderer()
			renderWin = vtk.vtkRenderWindow()
			renderWin.SetOffScreenRendering(1)

			renderWin.AddRenderer(renderer)
			renderInteractor = vtk.vtkRenderWindowInteractor()
			renderInteractor.SetRenderWindow(renderWin)

			renderer.AddVolume(volume_r)
			renderer.SetBackground((255,255,255))

			viewUp = [0, 1, 0]
			normal = [0,0,-1]

		elif orientation == 'left_medial':
			renderer = vtk.vtkRenderer()
			renderWin = vtk.vtkRenderWindow()
			renderWin.SetOffScreenRendering(1)

			renderWin.AddRenderer(renderer)
			renderInteractor = vtk.vtkRenderWindowInteractor()
			renderInteractor.SetRenderWindow(renderWin)

			renderer.AddVolume(volume_l)
			renderer.SetBackground((255,255,255))

			viewUp = [0, 1, 0]
			normal = [0, 0, 1]

		elif orientation == 'bg_t_axial':
			renderer = vtk.vtkRenderer()
			renderWin = vtk.vtkRenderWindow()
			renderWin.SetOffScreenRendering(1)

			renderWin.AddRenderer(renderer)
			renderInteractor = vtk.vtkRenderWindowInteractor()
			renderInteractor.SetRenderWindow(renderWin)

			renderer.AddVolume(volume_bg_t)
			renderer.SetBackground((255,255,255))

			viewUp = [0, 0, 1]
			normal = [0, -1, 0]

		renderer.ResetCamera()
		camera =  renderer.GetActiveCamera()
		focus = camera.GetFocalPoint()
		d = camera.GetDistance()

		camera.SetPosition(focus[0] + d*normal[0], focus[1] + d*normal[1], focus[2] + d*normal[2])
		camera.SetFocalPoint(focus)
		camera.SetViewUp(viewUp)
		camera.OrthogonalizeViewUp()

		vtk_show(orientation, fname, renderer, 800, 800)

	# Get colorbar and combine 4 images to get final image!!!
	img_lhl = cv2.imread(fname+'_lefthemosphere_lateral.png')
	img_rhl = cv2.imread(fname+'_righthemosphere_lateral.png')
	img_lm = cv2.imread(fname+'_left_medial.png')
	img_rm = cv2.imread(fname+'_right_medial.png')
	img_axialt = cv2.imread(fname+'_axialtop.png')
	img_axialb = cv2.imread(fname+'_axialbottom.png')
	img_bg_t_axial = cv2.imread(fname+'_bg_t_axial.png')

	# Alter spacing and overall image size here
	x_dim_common = int(max(img_lhl.shape[0],img_rhl.shape[0],img_lm.shape[0],img_rm.shape[0],img_axialt.shape[0],img_axialb.shape[0],img_bg_t_axial.shape[0])*1.3)
	y_dim_common = int(max(img_lhl.shape[1],img_rhl.shape[1],img_lm.shape[1],img_rm.shape[1],img_axialt.shape[1],img_axialb.shape[1],img_bg_t_axial.shape[1])*1.3)

	fig = matplotlib.pyplot.figure(figsize=((x_dim_common)/100,y_dim_common/100))
	# X, Y, width, height
	ax = fig.add_axes([0.05, 0.80, 0.05, 0.9])

	#cmap = matplotlib.colors.ListedColormap(['#882255','#aa4499','#cc6677','#ddcc77','#88ccee','#44aa99','#117733','#332288'])
	cmap = matplotlib.colors.ListedColormap(['#bd0026','#f03b20','#fd8d3c','#fecc5c','#ffffb2'])
	bounds=[min_zscore-1,min_zscore,-1.881,-1.645,-1.036,-.5244]
	ticks=[min_zscore,-1.881,-1.645,-1.036,-.5244]
	norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

	cb = matplotlib.colorbar.ColorbarBase(ax, orientation='vertical', 
										cmap = cmap,
										norm = norm)#,
										#ticks = ticks,
										#boundaries = bounds)

	cb.set_label('Z-score map', labelpad=15, rotation=270)
	cb.ax.set_yticklabels(['-∞','-2.326','-1.881','-1.645','-1.036','-.5244'])

	matplotlib.pyplot.savefig(fname+'_colorbar.png',bbox_inches='tight')

	final_outimg = np.full((x_dim_common*2,y_dim_common*4,3), (255,255,255), dtype=np.uint8)

	#### Place image arrays into larger array (leads to combined picture export) ####
	# Right Hemisphere Lateral
	spcr_x = int((x_dim_common-img_rhl.shape[0])/2)
	spcr_y = int((y_dim_common-img_rhl.shape[1])/2)
	final_outimg[0:x_dim_common, 0:y_dim_common] = np.pad(img_rhl,((spcr_x,x_dim_common-img_rhl.shape[0]-spcr_x),(spcr_y,y_dim_common-img_rhl.shape[1]-spcr_y),(0,0)),'constant', constant_values=255)

	# Right Hemisphere Medial
	spcr_x = int((x_dim_common-img_rm.shape[0])/2)
	spcr_y = int((y_dim_common-img_rm.shape[1])/2)
	final_outimg[0:x_dim_common, y_dim_common:y_dim_common*2] = np.pad(img_rm,((spcr_x,x_dim_common-img_rm.shape[0]-spcr_x),(spcr_y,y_dim_common-img_rm.shape[1]-spcr_y),(0,0)),'constant', constant_values=255)

	# Left Hemisphere Lateral
	spcr_x = int((x_dim_common-img_lhl.shape[0])/2)
	spcr_y = int((y_dim_common-img_lhl.shape[1])/2)
	final_outimg[0:x_dim_common, y_dim_common*2:y_dim_common*3] = np.pad(img_lhl,((spcr_x,x_dim_common-img_lhl.shape[0]-spcr_x),(spcr_y,y_dim_common-img_lhl.shape[1]-spcr_y),(0,0)),'constant', constant_values=255)

	# Left Hemisphere Medial
	spcr_x = int((x_dim_common-img_lm.shape[0])/2)
	spcr_y = int((y_dim_common-img_lm.shape[1])/2)
	final_outimg[0:x_dim_common, y_dim_common*3:y_dim_common*4] = np.pad(img_lm,((spcr_x,x_dim_common-img_lm.shape[0]-spcr_x),(spcr_y,y_dim_common-img_lm.shape[1]-spcr_y),(0,0)),'constant', constant_values=255)

	# Axial bottom
	spcr_x = int((x_dim_common-img_axialb.shape[0])/2)
	spcr_y = int((y_dim_common-img_axialb.shape[1])/2)
	final_outimg[x_dim_common:x_dim_common*2, 0:y_dim_common] = np.pad(img_axialb,((spcr_x,x_dim_common-img_axialb.shape[0]-spcr_x),(spcr_y,y_dim_common-img_axialb.shape[1]-spcr_y),(0,0)),'constant', constant_values=255)

	# Axial top
	spcr_x = int((x_dim_common-img_axialt.shape[0])/2)
	spcr_y = int((y_dim_common-img_axialt.shape[1])/2)
	final_outimg[x_dim_common:x_dim_common*2, y_dim_common:y_dim_common*2] = np.pad(img_axialt,((spcr_x,x_dim_common-img_axialt.shape[0]-spcr_x),(spcr_y,y_dim_common-img_axialt.shape[1]-spcr_y),(0,0)),'constant', constant_values=255)

	# Axial slice of basal ganglia and thalamus
	spcr_x = int((x_dim_common-img_bg_t_axial.shape[0])/2)
	spcr_y = int((y_dim_common-img_bg_t_axial.shape[1])/2)
	final_outimg[x_dim_common:x_dim_common*2, y_dim_common*2:y_dim_common*3] = np.pad(img_bg_t_axial,((spcr_x,x_dim_common-img_bg_t_axial.shape[0]-spcr_x),(spcr_y,y_dim_common-img_bg_t_axial.shape[1]-spcr_y),(0,0)),'constant', constant_values=255)

	# Add colorbar to edge of image array
	cb = cv2.imread(fname+'_colorbar.png')
	#spcr_x = int((x_dim_common*2-cb.shape[0])/2)
	#padded_cb = np.pad(cb,((spcr_x,x_dim_common*2-cb.shape[0]-spcr_x),(0,0),(0,0)),'constant', constant_values=255)
	spcr_x = int((x_dim_common-cb.shape[0])/2)
	spcr_y = int((y_dim_common-cb.shape[1])/2)
	#padded_cb = np.pad(cb,((spcr_x,x_dim_common*2-cb.shape[0]-spcr_x),(0,0),(0,0)),'constant', constant_values=255)
	padded_cb = np.pad(cb,((spcr_x,x_dim_common-cb.shape[0]-spcr_x),(spcr_y,y_dim_common-cb.shape[1]-spcr_y),(0,0)),'constant', constant_values=255)

	#final_outimg = np.pad(final_outimg,((0,0),(0,padded_cb.shape[1]),(0,0)),'constant',constant_values=255)
	#final_outimg[:,y_dim_common*4:,:] = padded_cb
	final_outimg[x_dim_common:x_dim_common*2, y_dim_common*3:y_dim_common*4] = padded_cb

	# Save array as image
	cv2.imwrite(fname+'_finalvis.png', final_outimg)

	img = Image.open(fname+'_finalvis.png')
	I1 = ImageDraw.Draw(img)

	# Declare font style and size
	myFont = ImageFont.truetype("refs/Times New Roman Bold.ttf", 25)
	text_width_rhl,text_height_rhl = I1.textsize("Right hemisphere lateral",myFont)
	text_width_rm,text_height_rm = I1.textsize("Right hemisphere medial",myFont)
	text_width_axialb,text_height_axialb = I1.textsize("Bottom",myFont)
	text_width_lhl,text_height_lhl = I1.textsize("Left hemisphere lateral",myFont)
	text_width_lm,text_height_lm = I1.textsize("Left hemisphere medial",myFont)
	text_width_axialt,text_height_axialt = I1.textsize("Top",myFont)
	text_width_bg_t_axial,text_height_bg_t_axial = I1.textsize("Basal-Ganglia/Thalamus Slice",myFont)

	#Extract bounding boxes around each image to understand where to place text
	# rhl
	t,b,l,r = getbounds(final_outimg[0:x_dim_common, 0:y_dim_common])
	# rm
	t2,b2,l2,r2 = getbounds(final_outimg[0:x_dim_common, y_dim_common:y_dim_common*2])
	# lhl
	t3,b3,l3,r3 = getbounds(final_outimg[0:x_dim_common, y_dim_common*2:y_dim_common*3])
	# lm
	t4,b4,l4,r4 = getbounds(final_outimg[0:x_dim_common, y_dim_common*3:y_dim_common*4])
	# axialb
	t5,b5,l5,r5 = getbounds(final_outimg[x_dim_common:x_dim_common*2, 0:y_dim_common])
	# axialt
	t6,b6,l6,r6 = getbounds(final_outimg[x_dim_common:x_dim_common*2, y_dim_common:y_dim_common*2])
	# bg_t_axial
	t7,b7,l7,r7 = getbounds(final_outimg[x_dim_common:x_dim_common*2, y_dim_common*2:y_dim_common*3])

	# Align first row of text - works
	text_ypos_row1 = min((t-text_height_rhl)/2,(t2-text_height_rm)/2,(t3-text_height_lhl)/2,(t4-text_height_lm)/2)
	# Align second row of text - works
	text_ypos_row2 = min((t5-text_height_axialb)/2,(t6-text_height_axialt)/2,(t7-text_height_bg_t_axial)/2) + x_dim_common
	# Get common L and R label positions for second row - works
	text_ypos_row2sub = max(b5 + (x_dim_common-b5-text_height_axialb)/2 + x_dim_common,b6 + (x_dim_common-b6-text_height_axialt)/2 + x_dim_common)

	# Align columns
	text_xpos_col1 = max(l + ((r-l)-text_width_rhl)/2,l5 + ((r5-l5)-text_width_axialb)/2)
	text_xpos_col2 = max(y_dim_common + l2 + ((r2-l2)-text_width_rm)/2,y_dim_common + l6 + ((r6-l6)-text_width_axialt)/2)
	text_xpos_col3 = max(y_dim_common*2 + l3 + ((r3-l3)-text_width_lhl)/2,y_dim_common*2 + l7 + ((r7-l7)-text_width_bg_t_axial)/2)
	text_xpos_col4 = y_dim_common*3 + l4 + ((r4-l4)-text_width_lm)/2

	# Align L and R text for top and bottom axial views
	lmin = min(l5,l6)
	rmax = max(r5,r6)

	# Row 1 positions
	pos_rhl = (l + ((r-l)-text_width_rhl)/2,text_ypos_row1)
	pos_rm = (l2 + x_dim_common + ((r2-l2)-text_width_rm)/2,text_ypos_row1)
	pos_lhl = (l3 + x_dim_common*2 + ((r3-l3)-text_width_lhl)/2,text_ypos_row1)
	pos_lm = (l4 + x_dim_common*3 + ((r4-l4)-text_width_lm)/2,text_ypos_row1)

	# Row 2 positions, while making sure to line up with Row 1 positions vertically
	# Unique code to align by itself
	pos_axialb = (l5 + ((r5-l5)-text_width_axialb)/2,text_ypos_row2)
	###
	pos_row2_axialb_l = (lmin,text_ypos_row2sub)
	pos_row2_axialb_r = (rmax,text_ypos_row2sub)

	# Unique code to align by itself
	pos_axialt = (y_dim_common + l6 + ((r6-l6)-text_width_axialt)/2,text_ypos_row2)
	###
	pos_row2_axialt_l = (y_dim_common + lmin,text_ypos_row2sub)
	pos_row2_axialt_r = (y_dim_common + rmax,text_ypos_row2sub)
	pos_bg_t_axial = (y_dim_common*2 + l7 + ((r7-l7)-text_width_bg_t_axial)/2,text_ypos_row2)

	I1.text(pos_rhl, "Right hemisphere lateral", font = myFont, fill=(0, 0, 0))
	I1.text(pos_rm, "Right hemisphere medial", font = myFont, fill=(0, 0, 0))
	I1.text(pos_axialb, "Bottom", font = myFont, fill=(0, 0, 0))
	I1.text(pos_lhl, "Left hemisphere lateral", font = myFont, fill=(0, 0, 0))
	I1.text(pos_lm, "Left hemisphere medial", font = myFont, fill=(0, 0, 0))
	I1.text(pos_axialt, "Top", font = myFont, fill=(0, 0, 0))
	I1.text(pos_bg_t_axial, "Basal-Ganglia/Thalamus Slice", font = myFont, fill=(0, 0, 0))
	I1.text(pos_row2_axialb_l, "R", font = myFont, fill=(0, 0, 0))
	I1.text(pos_row2_axialb_r, "L", font = myFont, fill=(0, 0, 0))
	I1.text(pos_row2_axialt_l, "L", font = myFont, fill=(0, 0, 0))
	I1.text(pos_row2_axialt_r, "R", font = myFont, fill=(0, 0, 0))

	# Final image save
	img.save(fname+'_finalvis.png')

def _main( roi, allz_num, pdf_path):
	UID = _os.path.basename(pdf_path.removesuffix(".pdf"))
	out = _os.path.dirname(pdf_path)
	out = out + '/' + UID

	#_os.environ['DISPLAY'] =':99.0'

	#commands = ['Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &','sleep 3','exec "$@"']

	#for command in commands:
	#	subprocess.call(command,shell=True)

	atrophyvisualization(roi,allz_num,out)