import os as _os
import numpy as np
#import cv2

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

def WMHbox(fname):
	# Add 16 pixels to top of image and 35 to right of image to allow box to be drawn
	# plot = cv2.imread(fname)
	# padded_plot = np.pad(plot,((16,0),(0,35),(0,0)),'constant',constant_values=255)
	# cv2.imwrite(fname, padded_plot)
	img = Image.open(fname)
	plot = np.array(img)
	padding_values = ((16, 0), (0, 35), (0, 0))
	padded_plot = np.pad(plot, padding_values, 'constant', constant_values=255)
	padded_img = Image.fromarray(padded_plot)
	padded_img.save(fname)

	img = Image.open(fname)
	I1 = ImageDraw.Draw(img)

	I1.line((1451,12, 1451,467), width=2, fill=128)
	I1.line((1451,467, 1938,467), width=2, fill=128)
	I1.line((1451,12, 1938,12), width=2, fill=128)
	I1.line((1938,12, 1938,467), width=2, fill=128)

	myFont = ImageFont.truetype("/refs/Times New Roman Bold.ttf", 14)

	# Write 'Unharmonized values' in lower left of the box
	I1.text((1454,450), "Unharmonized population values", font = myFont, fill=128)

	img.save(fname)