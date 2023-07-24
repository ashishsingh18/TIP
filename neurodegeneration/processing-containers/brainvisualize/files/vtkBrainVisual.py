import os as _os
# import cv2
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
from enum import Enum

#maphemi = pd.read_csv('/Users/vikasbommineni/Desktop/MRIreport/brainvisualize/refs/MUSE_ROI_Dictionary.csv')
maphemi = pd.read_csv('../refs/MUSE_ROI_Dictionary.csv')

class Orientation(Enum):
	Right_hemisphere_lateral = 1
	Right_hemisphere_medial = 2
	Left_hemisphere_lateral = 3
	Left_hemisphere_medial = 4
	Bottom = 5
	Top = 6
	Basal_Ganglia_Thalamus = 7
	Colorbar = 8

title_top_row_from_left_1 = 'Right hemisphere lateral'
title_top_row_from_left_2 = 'Right hemisphere medial'
title_top_row_from_left_3 = 'Left hemisphere lateral'
title_top_row_from_left_4 = 'Left hemisphere medial'
title_bottom_row_from_left_1 = 'Bottom'
title_bottom_row_from_left_2 = 'Top'
title_bottom_row_from_left_3 = 'Basal-Ganglia/Thalamus'
title_bottom_row_from_left_4 = 'Colorbar'

def getbounds(arr):
	arr = np.mean(arr,axis=2)
	pos = np.nonzero(arr == 0)

	top = pos[0].min()
	bottom = pos[0].max()

	left = pos[1].min()
	right = pos[1].max()

	return top,bottom,left,right

# Crops each orientation image with 10 pixels boundary
# def crop_and_write(img,orientation,fname,x_dim,y_dim):
#     t,b,l,r = getbounds(img)

#     img = img[(t-10):(b+10),(l-10):(r+10)]
#     cv2.imwrite(fname+'_' + orientation + '.png',img)

def creat_title(renderer, text):
	# Create a title
	titleProperty = vtk.vtkTextProperty()
	titleProperty.SetFontSize(16)
	titleProperty.SetJustificationToCentered()

	titleMapper = vtk.vtkTextMapper()
	titleMapper.SetInput(text)
	titleMapper.SetTextProperty(titleProperty)

	titleActor = vtk.vtkActor2D()
	titleActor.SetMapper(titleMapper)
	titleActor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
	titleActor.GetPositionCoordinate().SetValue(0.5, 0.85, 0.0)
	titleActor.GetProperty().SetColor(vtk.vtkNamedColors().GetColor3d("Black"))

	# Add the titleActor to the renderer
	renderer.AddActor2D(titleActor)

def generate_slice_view(fname,relabelMap,need2relabel):

	reader_all = vtk.vtkNIFTIImageReader()
	reader_all.SetFileName(fname)
	castFilter_all = vtk.vtkImageCast()
	castFilter_all.SetInputConnection(reader_all.GetOutputPort())
	castFilter_all.SetOutputScalarTypeToUnsignedShort()
	castFilter_all.Update()
	imdataBrainSeg_all = castFilter_all.GetOutput()

	# lut = get_lut()
	lut = get_color_TF(relabelMap,need2relabel)
	scalarValuesToColors = vtk.vtkImageMapToColors()
	scalarValuesToColors.SetLookupTable(lut)
	# scalarValuesToColors.PassAlphaToOutputOn()
	scalarValuesToColors.SetInputData(imdataBrainSeg_all)


	imageViewer = vtk.vtkImageViewer2()
	imageViewer.SetInputConnection(castFilter_all.GetOutputPort())
	iren = vtk.vtkRenderWindowInteractor()
	imageViewer.SetupInteractor(iren)
	imageViewer.SetSlice(100)
	imageViewer.Render()
	imageViewer.GetRenderer().ResetCamera()
	iren.Initialize()
	imageViewer.Render()
	iren.Start()

def hex_to_rgb(hex: str):
	hex = hex[1:]
	assert len(hex) == 6
	# return [int(hex[i:i + 2], 16) for i in (0, 2, 4)]
	rgb = [int(hex[i:i + 2], 16) for i in (0, 2, 4)]
	rgb = [x/256 for x in rgb]
	print(rgb)
	return rgb

def get_lut():
	#lut
	min_zscore = -2.326
	colors = ['#bd0026','#f03b20','#fd8d3c','#fecc5c','#ffffb2']
	values = [min_zscore-1,min_zscore,-1.881,-1.645,-1.036,-.5244]
	nvals = len(colors)
	print("len: ", nvals)
	lut = vtk.vtkLookupTable()
	lut.SetNumberOfTableValues(nvals)
	lut.SetTableRange(min_zscore-1,-0.5244)	
	# lut.Build()
	i = 0
	for k in colors:
		rgba = hex_to_rgb(colors[i]) + [1] #[1] for opacity in rgba
		print("k: ", k, " rgb: ", rgba)
		lut.SetTableValue(i,rgba)
		lut.SetAnnotation(vtk.vtkVariant(i),str(values[i]))
		#print("k: ", k, " cmlabels: ", cmlabels[0][i])
		i = i + 1

	#exit()
	# lut.SetTableValue(0, 1.0, 0.0, 0.0) # red
	# lut.SetTableValue(1, 0.0, 1.0, 0.0) #blue
	# lut.SetTableValue(2, 1.0, 1.0, 1.0) # white
	# lut.SetAnnotation(vtk.vtkVariant(0),str(values[0]))
	# lut.SetAnnotation(vtk.vtkVariant(1),str(values[1]))
	# lut.SetAnnotation(vtk.vtkVariant(2),str(values[2]))

	# # # lut.SetIndexedLookup(1)
	lut.IndexedLookupOn()
	# lut.UseBelowRangeColorOn()
	# lut.UseAboveRangeColorOn()
	lut.SetBelowRangeColor(255,255,255,1)
	lut.SetAboveRangeColor(0,0,0,1)
	# lut.Build()
	return lut

def add_scalarbar(renderer):
	lut = get_lut()
	# print(lut)
	#scalar bar
	scalarBarActor = vtk.vtkScalarBarActor()
	scalarBarActor.SetLookupTable(lut)
	scalarBarActor.SetNumberOfLabels(lut.GetNumberOfAvailableColors())
	scalarBarActor.GetLabelTextProperty().ItalicOff()
	scalarBarActor.GetLabelTextProperty().BoldOff()
	scalarBarActor.GetLabelTextProperty().ShadowOff()
	scalarBarActor.GetAnnotationTextProperty().SetFontSize(20)
	scalarBarActor.GetLabelTextProperty().SetColor(1.0,0.0,0.0)
	scalarBarActor.SetLabelFormat('%.2f')
	renderer.AddActor(scalarBarActor)

def render_scalarbar():
	rw = vtk.vtkRenderWindow()
	iren = vtk.vtkRenderWindowInteractor()
	iren.SetRenderWindow(rw)
	ren = vtk.vtkRenderer()
	rw.AddRenderer(ren)
	ren.SetBackground(0,0,1)
	add_scalarbar(ren)

	rw.Render()
	rw.SetWindowName('scalarbar')
	# rw.SetSize(600, 600)

	iren.Start()

def get_color_TF(relabelMap,need2relabel):
	funcColor = vtk.vtkColorTransferFunction()

	print('relabelmap: ', relabelMap)
	print('need2relabel: ', need2relabel)

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
	return funcColor


def get_plane_from_mri(fname):
	muse_labelmap = sitk.ReadImage(fname)
	nda_bg_t = sitk.GetArrayFromImage(muse_labelmap)

	print('image shape is: ', nda_bg_t.shape)

	bg_t = ((nda_bg_t == 59) | (nda_bg_t == 60) | (nda_bg_t == 23) | (nda_bg_t == 30) | (nda_bg_t == 36) | (nda_bg_t == 37) | (nda_bg_t == 55) | (nda_bg_t == 56) | (nda_bg_t == 57) | (nda_bg_t == 58)).astype(int)
	print(bg_t.shape)
	cnt = np.inf
	bg_t_slice = 0
	# for y in range(bg_t.shape[1]):
	# 	if abs(np.sum(bg_t[:,:y,:]) - np.sum(bg_t[:,y:,:])) < cnt:
	# 		bg_t_slice = y
	# 		cnt = abs(np.sum(bg_t[:,:y,:]) - np.sum(bg_t[:,y:,:]))

	for x in range(bg_t.shape[0]):
		if abs(np.sum(bg_t[:x,:,:]) - np.sum(bg_t[x:,:,:])) < cnt:
			bg_t_slice = x
			cnt = abs(np.sum(bg_t[:x,:,:]) - np.sum(bg_t[x:,:,:]))
		
	# Set voxels to 0 above V.O.I
	# nda_bg_t[:,:bg_t_slice,:] = 0
	print('bgt slice: ', bg_t_slice)

	# Get transition point from left to right hemisphere from corpus callosum
	nda_all = nda_bg_t
	cc = (nda_all == 95).astype(int)

	#### test slices to get the most equitable cc split
	cnt = np.inf
	mid_slice = 0
	for y in range(cc.shape[1]):
		if abs(np.sum(cc[:,:y,:]) - np.sum(cc[:,y:,:])) < cnt:
			mid_slice = y
			cnt = abs(np.sum(cc[:,:y,:]) - np.sum(cc[:,y:,:]))

	print('mid slice: ', mid_slice)

	return bg_t_slice, mid_slice


def get_volume(fname,relabelMap,need2relabel,clip, orientation, *args):
	
	if len(args) != 0:
		bg_t_slice, mid_slice = args #get_plane_from_mri(fname)

	# for i in range(len(list_filenames)):
	# fname = list_filenames[i]
	reader_all = vtk.vtkNIFTIImageReader()
	reader_all.SetFileName(fname)
	castFilter_all = vtk.vtkImageCast()
	castFilter_all.SetInputConnection(reader_all.GetOutputPort())
	castFilter_all.SetOutputScalarTypeToUnsignedShort()
	castFilter_all.Update()
	# print(castFilter_all.GetOutput().shape)
	# print("castFilter")
	# exit()
	imdataBrainSeg_all = castFilter_all.GetOutputPort()
	idbs_all = castFilter_all.GetOutput()

	# Define color legend #
	funcColor = vtk.vtkColorTransferFunction()

	PERCENTILE_1  = -2.326
	PERCENTILE_3  = -1.881
	PERCENTILE_6  = -1.645
	PERCENTILE_15 = -1.036
	PERCENTILE_30 = -0.5244

	for idx in relabelMap.keys():
		if idx in need2relabel:
			# TODO: create a fixed z-score based on most extreme of training data

			## below value chosen by Ilya as standard buckets
			if PERCENTILE_15 <= relabelMap[idx] <= PERCENTILE_30:   
				funcColor.AddRGBPoint(idx,255/255,255/255,178/255)
			elif PERCENTILE_6 <= relabelMap[idx] < PERCENTILE_15: 
				funcColor.AddRGBPoint(idx,254/255,204/255,92/255)
			elif PERCENTILE_3 <= relabelMap[idx] < PERCENTILE_6:  
				funcColor.AddRGBPoint(idx,253/255,141/255,60/255)
			elif PERCENTILE_1 <= relabelMap[idx] < PERCENTILE_3:  
				funcColor.AddRGBPoint(idx,240/255,59/255,32/255)
			elif relabelMap[idx] < PERCENTILE_1: 
				funcColor.AddRGBPoint(idx,189/255,0/255,38/255)
		else:
			funcColor.AddRGBPoint(idx,1,1,1)

	funcColor.AddRGBPoint(0,1,1,1)

	# Define opacity scheme #
	funcOpacityScalar = vtk.vtkPiecewiseFunction()
	gradopacity = vtk.vtkPiecewiseFunction()
	funcOpacityScalar.AddPoint(0, 0)
	for idx in relabelMap.keys():
		funcOpacityScalar.AddPoint(idx, 1 if idx > 0 else 0.0)
		gradopacity.AddPoint(idx, 1 if idx > 0 else 0.0)
	gradopacity.AddPoint(0, 0)

	### WHOLE VOLUME ###
	propVolume_all = vtk.vtkVolumeProperty()
	propVolume_all.SetColor(funcColor)
	propVolume_all.SetScalarOpacity(funcOpacityScalar)
	propVolume_all.SetGradientOpacity(gradopacity)
	propVolume_all.SetInterpolationTypeToNearest()
	propVolume_all.ShadeOn()
	propVolume_all.SetDiffuse(0.7)
	propVolume_all.SetAmbient(0.8)
	volumeMapper_all = vtk.vtkFixedPointVolumeRayCastMapper()
	volumeMapper_all.SetInputConnection(imdataBrainSeg_all)
	if(clip):
		plane = vtk.vtkPlane()
		if(orientation == title_top_row_from_left_2):

			print('left 2', idbs_all.GetCenter())

			plane = vtk.vtkPlane()
			origin = list(idbs_all.GetCenter())
			origin[1] = mid_slice

			print('origin', origin)
			plane.SetOrigin(origin)
			plane.SetNormal(-1,0,0)

			# plane = vtk.vtkPlane()
			# plane.SetOrigin(idbs_all.GetCenter()) #[x, y, z]
			# plane.SetNormal(-1,0,0)
		elif(orientation == title_top_row_from_left_4):
			print('left 4', idbs_all.GetCenter())

			plane = vtk.vtkPlane()
			origin = list(idbs_all.GetCenter())
			origin[1] = mid_slice

			print('origin', origin)
			plane.SetOrigin(origin)
			plane.SetNormal(1,0,0)

			# plane = vtk.vtkPlane()
			# plane.SetOrigin(idbs_all.GetCenter())
			# plane.SetNormal(1,0,0)
		elif(orientation == title_bottom_row_from_left_3):

			print('left 3', idbs_all.GetCenter())

			plane = vtk.vtkPlane()
			origin = list(idbs_all.GetCenter())
			origin[2] = bg_t_slice

			print('origin', origin)
			plane.SetOrigin(origin)
			plane.SetNormal(0,0,-1)
			# plane = vtk.vtkPlane()
			# plane.SetOrigin(idbs_all.GetCenter())
			# plane.SetNormal(0,0,-1)
		volumeMapper_all.AddClippingPlane(plane)

	volume_all = vtk.vtkVolume()
	volume_all.SetMapper(volumeMapper_all)
	volume_all.SetProperty(propVolume_all)
	return volume_all

# def get_clipped_volume(fname,relabelMap,need2relabel):
# 	# for i in range(len(list_filenames)):
# 	# fname = list_filenames[i]
# 	reader_all = vtk.vtkNIFTIImageReader()
# 	reader_all.SetFileName(fname)
# 	castFilter_all = vtk.vtkImageCast()
# 	castFilter_all.SetInputConnection(reader_all.GetOutputPort())
# 	castFilter_all.SetOutputScalarTypeToUnsignedShort()
# 	castFilter_all.Update()
# 	# print(castFilter_all.GetOutput())
# 	# print("castFilter")
# 	# exit()
# 	imdataBrainSeg_all = castFilter_all.GetOutputPort()
# 	idbs_all = castFilter_all.GetOutput()

# 	# Define color legend #
# 	funcColor = vtk.vtkColorTransferFunction()

# 	for idx in relabelMap.keys():
# 		if idx in need2relabel:
# 			# TODO: create a fixed z-score based on most extreme of training data
# 			if -1.036 <= relabelMap[idx] <= -.5244:
# 				funcColor.AddRGBPoint(idx,255/255,255/255,178/255)
# 			elif -1.645 <= relabelMap[idx] < -1.036:
# 				funcColor.AddRGBPoint(idx,254/255,204/255,92/255)
# 			elif -1.881 <= relabelMap[idx] < -1.645:
# 				funcColor.AddRGBPoint(idx,253/255,141/255,60/255)
# 			elif -2.326 <= relabelMap[idx] < -1.881:
# 				funcColor.AddRGBPoint(idx,240/255,59/255,32/255)
# 			elif relabelMap[idx] < -2.326:
# 				funcColor.AddRGBPoint(idx,189/255,0/255,38/255)
# 		else:
# 			funcColor.AddRGBPoint(idx,1,1,1)

# 	funcColor.AddRGBPoint(0,1,1,1)

# 	# Define opacity scheme #
# 	funcOpacityScalar = vtk.vtkPiecewiseFunction()
# 	funcOpacityScalar.AddPoint(0, 0)
# 	for idx in relabelMap.keys():
# 		funcOpacityScalar.AddPoint(idx, 1 if idx > 0 else 0.0)

# 	#create plane
# 	plane = vtk.vtkPlane()
# 	#plane.SetOrigin(-2.78,15.13,-4.41)
# 	plane.SetOrigin(idbs_all.GetCenter())
# 	plane.SetNormal(-1,0,0)

# 	### WHOLE VOLUME ###
# 	propVolume_all = vtk.vtkVolumeProperty()
# 	propVolume_all.SetColor(funcColor)
# 	propVolume_all.SetScalarOpacity(funcOpacityScalar)
# 	propVolume_all.SetInterpolationTypeToNearest()
# 	propVolume_all.ShadeOn()
# 	propVolume_all.SetDiffuse(0.7)
# 	propVolume_all.SetAmbient(0.8)
# 	volumeMapper_all = vtk.vtkSmartVolumeMapper()
# 	volumeMapper_all.SetInputConnection(imdataBrainSeg_all)
# 	volumeMapper_all.AddClippingPlane(plane)

# 	volume_all = vtk.vtkVolume()
# 	volume_all.SetMapper(volumeMapper_all)
# 	volume_all.SetProperty(propVolume_all)
# 	return volume_all


def save_screeenshot(rw,filename):
	windowToImageFilter = vtk.vtkWindowToImageFilter()
	windowToImageFilter.SetInput(rw)
	windowToImageFilter.Update()

	writer = vtk.vtkPNGWriter()
	writer.SetFileName(filename)
	#writer.SetWriteToMemory(1)
	writer.SetInputConnection(windowToImageFilter.GetOutputPort())
	writer.Write()

def setup_camera(orientation,renderer):
	# pass
	# cam = vtk.vtkcamera()
	if(orientation == title_bottom_row_from_left_2): #Top
		renderer.ResetCamera()
		camera =  renderer.GetActiveCamera()
		focus = camera.GetFocalPoint()
		d = camera.GetDistance()

		distance = camera.GetDistance()
		newdis = 0.75 * distance
		camera.SetPosition(focus[0],focus[1],(focus[2] + newdis))

		camera.SetFocalPoint(focus)
		camera.SetViewUp(0,-1,0)
		camera.OrthogonalizeViewUp()
		renderer.ResetCameraClippingRange()
	elif(orientation == title_bottom_row_from_left_1):#Bottom
		renderer.ResetCamera()
		camera =  renderer.GetActiveCamera()
		focus = camera.GetFocalPoint()
		d = camera.GetDistance()

		# spacing = [1,1,1]
		# extent = renderer.ComputeVisiblePropBounds()
		# imageHeight = (extent[3] - extent[2] + 1.0) * spacing[1]
		# viewAngleRadians = vtk.vtkMath.RadiansFromDegrees (camera.GetViewAngle())
		# distance = imageHeight / viewAngleRadians
		# print(viewAngleRadians)
		# camera.SetPosition(focus[0],focus[1],distance)
		# camera.SetPosition(0.5)

		distance = camera.GetDistance()
		newdis = 0.7 * distance
		camera.SetPosition(focus[0],focus[1],(focus[2] + newdis))

		camera.SetFocalPoint(focus)
		camera.SetViewUp(0,-1,0)
		camera.Azimuth(180)
		camera.OrthogonalizeViewUp()
		renderer.ResetCameraClippingRange()
	if(orientation == title_bottom_row_from_left_3): #Basal Ganglia/Thalamus
		renderer.ResetCamera()
		camera =  renderer.GetActiveCamera()
		focus = camera.GetFocalPoint()
		d = camera.GetDistance()

		distance = camera.GetDistance()
		newdis = 0.75 * distance
		camera.SetPosition(focus[0],focus[1],focus[2]+newdis)

		camera.SetFocalPoint(focus)
		camera.SetViewUp(0,-1,0)
		camera.OrthogonalizeViewUp()
		renderer.ResetCameraClippingRange()
	if(orientation == title_top_row_from_left_1): #Right hemisphere lateral
		renderer.ResetCamera()
		camera =  renderer.GetActiveCamera()
		focus = camera.GetFocalPoint()
		d = camera.GetDistance()

		distance = camera.GetDistance()
		newdis = 0.75 * distance
		camera.SetPosition((focus[0] + newdis),focus[1],focus[2])

		camera.SetFocalPoint(focus)
		camera.SetViewUp(0,0,1)
		# camera.Roll(90)
		# camera.
		camera.OrthogonalizeViewUp()
		renderer.ResetCameraClippingRange()
	if(orientation == title_top_row_from_left_2): #Right hemisphere medial
		renderer.ResetCamera()
		camera =  renderer.GetActiveCamera()
		focus = camera.GetFocalPoint()
		d = camera.GetDistance()

		distance = camera.GetDistance()
		newdis = 0.7 * distance
		camera.SetPosition(focus[0],focus[1],focus[2]+newdis)
		camera.SetPosition(541.09,15.13,-4.41)
		#camera.SetPosition(focus[0],focus[1]+newdis,focus[2])
		

		camera.SetFocalPoint(focus)
		camera.SetViewUp(0,0,1)
		camera.OrthogonalizeViewUp()
		renderer.ResetCameraClippingRange()

	if(orientation == title_top_row_from_left_3): #Left hemisphere lateral
		renderer.ResetCamera()
		camera =  renderer.GetActiveCamera()
		focus = camera.GetFocalPoint()
		d = camera.GetDistance()

		distance = camera.GetDistance()
		newdis = 0.55 * distance
		camera.SetPosition(-(focus[0] + newdis),focus[1],focus[2])

		camera.SetFocalPoint(focus)
		camera.SetViewUp(0,0,1)
		camera.OrthogonalizeViewUp()
		renderer.ResetCameraClippingRange()
	if(orientation == title_top_row_from_left_4): #Left hemisphere medial
		renderer.ResetCamera()
		camera =  renderer.GetActiveCamera()
		focus = camera.GetFocalPoint()
		d = camera.GetDistance()

		distance = camera.GetDistance()
		newdis = 0.6 * distance
		camera.SetPosition(-(focus[0] + newdis),focus[1],focus[2])

		camera.SetFocalPoint(focus)
		camera.SetViewUp(0,0,1)
		camera.OrthogonalizeViewUp()
		renderer.ResetCameraClippingRange()	


def setup_vtk_pipeline(roi,allz_num,out):

	relabelMap = get_relabel_map(roi,allz_num)

	list_filenames,need2relabel = generate_data_for_visualization(relabelMap,roi,allz_num,out)

	# generate_slice_view(list_filenames['all_filenameSegmentation'],relabelMap,need2relabel)
	# exit()
	# print(list_renderers.keys())
	colors = vtk.vtkNamedColors()
	# One render window, multiple viewports.
	rw = vtk.vtkRenderWindow()
	iren = vtk.vtkRenderWindowInteractor()
	iren.SetRenderWindow(rw)

	# Define viewport ranges.
	viewport_positions = {}
	viewport_positions[title_bottom_row_from_left_1] = [0, 0, 0.25, 0.5]
	viewport_positions[title_bottom_row_from_left_2] = [0.25, 0, 0.5, 0.5]
	viewport_positions[title_bottom_row_from_left_3] = [0.5, 0, 0.75, 0.5]
	viewport_positions[title_bottom_row_from_left_4] = [0.75, 0, 1, 0.5]
	viewport_positions[title_top_row_from_left_1] = [0, 0.5, 0.25, 1]
	viewport_positions[title_top_row_from_left_2] = [0.25, 0.5, 0.5, 1]
	viewport_positions[title_top_row_from_left_3] = [0.5, 0.5, 0.75, 1]
	viewport_positions[title_top_row_from_left_4] = [0.75, 0.5, 1, 1]

	# Have some fun with colors.
	ren_bkg = ['blue', 'pink', 'yellow', 'brown', 'green', 'red','purple','cyan']
	# actor_color = ['Bisque', 'RosyBrown', 'Goldenrod', 'Chocolate']
	# renderers = list(list_renderers.values())

	#create all renderers
	viewport_keys = list(viewport_positions.keys())
	list_renderers = {}
	for i in range(8):
		ren = vtk.vtkRenderer()
		rw.AddRenderer(ren)
		pos = viewport_positions[viewport_keys[i]]
		# ren.SetWindowName(viewport_keys[i])
		ren.SetViewport(pos[0], pos[1], pos[2], pos[3])
		#ren.SetBackground(colors.GetColor3d(ren_bkg[i]))
		ren.SetBackground(1,1,1)
		ren.ResetCamera()
		list_renderers[viewport_keys[i]] = ren
		
	####################### I add some codes here ##########################

	########################################################################

	#add data to renderers

	bg_t_slice, mid_slice = get_plane_from_mri(list_filenames['all_filenameSegmentation'])

	for i in  range(len(list_renderers)):
	#	pass
		#print(list_renderers.keys())
		# print(list_renderers.get(viewport_keys[i]))
		print(viewport_keys[i])
		ren = list_renderers[viewport_keys[i]]
		if(viewport_keys[i] == title_top_row_from_left_1):
			volume = get_volume(list_filenames['all_filenameSegmentation'],relabelMap,need2relabel,False,title_top_row_from_left_1)
			ren.AddVolume(volume)
			creat_title(ren,title_top_row_from_left_1)
			setup_camera(title_top_row_from_left_1,ren)
		if(viewport_keys[i] == title_top_row_from_left_3):
			volume = get_volume(list_filenames['all_filenameSegmentation'],relabelMap,need2relabel,False,title_top_row_from_left_3)
			ren.AddVolume(volume)
			creat_title(ren,title_top_row_from_left_3)
			setup_camera(title_top_row_from_left_3,ren)
		if(viewport_keys[i] == title_bottom_row_from_left_1):
			volume = get_volume(list_filenames['all_filenameSegmentation'],relabelMap,need2relabel,False,title_bottom_row_from_left_1)
			ren.AddVolume(volume)
			creat_title(ren,title_bottom_row_from_left_1)
			setup_camera(title_bottom_row_from_left_1,ren)
		if(viewport_keys[i] == title_bottom_row_from_left_2):
			volume = get_volume(list_filenames['all_filenameSegmentation'],relabelMap,need2relabel,False,title_bottom_row_from_left_2)
			ren.AddVolume(volume)
			creat_title(ren,title_bottom_row_from_left_2)
			setup_camera(title_bottom_row_from_left_2,ren)
		elif(viewport_keys[i] == title_top_row_from_left_2):
			#volume = get_volume(list_filenames['r_filenameSegmentation'],relabelMap,need2relabel)
			#volume = get_volume(list_filenames['all_filenameSegmentation'],relabelMap,need2relabel)
			volume = get_volume(list_filenames['all_filenameSegmentation'],relabelMap,need2relabel,True,title_top_row_from_left_2, bg_t_slice, mid_slice)
			ren.AddVolume(volume)
			creat_title(ren,title_top_row_from_left_2)
			setup_camera(title_top_row_from_left_2,ren)
		elif(viewport_keys[i] == title_top_row_from_left_4):
			# volume = get_volume(list_filenames['l_filenameSegmentation'],relabelMap,need2relabel,True,title_top_row_from_left_4)
			volume = get_volume(list_filenames['all_filenameSegmentation'],relabelMap,need2relabel,True,title_top_row_from_left_4, bg_t_slice, mid_slice)
			ren.AddVolume(volume)
			creat_title(ren,title_top_row_from_left_4)
			setup_camera(title_top_row_from_left_4,ren)
		elif(viewport_keys[i] == title_bottom_row_from_left_3):
			# volume = get_volume(list_filenames['bg_t_filenameSegmentation'],relabelMap,need2relabel,True,title_bottom_row_from_left_3)
			volume = get_volume(list_filenames['all_filenameSegmentation'],relabelMap,need2relabel,True,title_bottom_row_from_left_3, bg_t_slice, mid_slice)
			ren.AddVolume(volume)
			creat_title(ren,title_bottom_row_from_left_3)
			setup_camera(title_bottom_row_from_left_3,ren)
		elif(viewport_keys[i] == title_bottom_row_from_left_4): #bottom row, last renderer for colorbar
			add_scalarbar(ren)
			# pass
				

	# 	#setup camera for render
		# if(viewport_keys[i] == title_bottom_row_from_left_1):
			
	# elif

	#setup camera orientation on renderers

	rw.Render()
	rw.SetWindowName('MultipleViewPorts')
	# rw.SetSize(600, 600)

	save_screeenshot(rw,"all.png")

	iren.Start()


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

def get_relabel_map(maskfile, allz):
	# Path to the file
	filenameSegmentation = maskfile
	muse_labelmap=readimage(filenameSegmentation)
	roi_zscore_dict=allz
	relabelMap = create_relabel_map(muse_labelmap,roi_zscore_dict)
	return relabelMap

def generate_data_for_visualization(relabelMap,maskfile,allz,fname):
	muse_labelmap=readimage(maskfile)
	roi_zscore_dict=allz
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
	# print(nda_all.shape)
	# print(nda_l.shape)
	# print(nda_r.shape)
	# print(nda_bg_t.shape)
	
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
	# print('bgt slice: ', bg_t_slice)

	# Get transition point from left to right hemisphere from corpus callosum
	cc = (nda_all == 95).astype(int)

	#### test slices to get the most equitable cc split
	cnt = np.inf
	mid_slice = 0
	for x in range(cc.shape[0]):
		if abs(np.sum(cc[x:,:,:])-np.sum(cc[:x,:,:])) < cnt:
			mid_slice = x
			cnt = abs(np.sum(cc[x:,:,:])-np.sum(cc[:x,:,:]))
	# print('mid slice: ', mid_slice)

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
	
	list_filenames = {}
	list_filenames['all_filenameSegmentation'] = all_filenameSegmentation
	list_filenames['l_filenameSegmentation'] = l_filenameSegmentation
	list_filenames['r_filenameSegmentation'] = r_filenameSegmentation
	list_filenames['bg_t_filenameSegmentation'] = bg_t_filenameSegmentation

	# print(174)

	write_image(all_post,all_filenameSegmentation)
	write_image(l_post,l_filenameSegmentation)
	write_image(r_post,r_filenameSegmentation)
	write_image(bg_t_post,bg_t_filenameSegmentation)

	return (list_filenames,need2relabel)

def _main( roi, allz_num, pdf_path):
	UID = _os.path.basename(pdf_path.removesuffix(".pdf"))
	out = _os.path.dirname(pdf_path)
	out = out + '/' + UID

	print('out: ', out)
	print(590)

	setup_vtk_pipeline(roi,allz_num,out)

if __name__ == '__main__':
	# print(599)
	roi = '/home/diwu/Desktop/F2/2.16.840.1.114362.1.12066432.24920037488.604832326.447.1607/relabel/2.16.840.1.114362.1.12066432.24920037488.604832326.447.1607.nii.gz'
	input = readimage(roi)
	output = write_image(input,'/home/diwu/Desktop/F2/2.16.840.1.114362.1.12066432.24920037488.604832326.447.1607/relabel/2.16.840.1.114362.1.12066432.24920037488.604832326.447.1607.nrrd')
	# print(603)
	roi_file = '/home/diwu/Desktop/F2/2.16.840.1.114362.1.12066432.24920037488.604832326.447.1607/relabel/2.16.840.1.114362.1.12066432.24920037488.604832326.447.1607.nrrd'
	with open('/home/diwu/Desktop/F2/2.16.840.1.114362.1.12066432.24920037488.604832326.447.1607/roi-quantification/2.16.840.1.114362.1.12066432.24920037488.604832326.447.1607_allz_num.pkl','rb') as f:
		allz_num = pickle.load(f)
	pdf_path = './test.pdf'
	_main( roi_file, allz_num, pdf_path)