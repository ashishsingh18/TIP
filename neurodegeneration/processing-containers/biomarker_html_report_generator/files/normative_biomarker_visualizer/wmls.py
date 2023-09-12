import pandas as pd
import numpy as np
import nrrd
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.linear_model import Ridge
from sklearn import metrics
from sklearn.base import clone
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import plotly as py
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from pygam import ExpectileGAM
from pygam.datasets import mcycle
from pygam import LinearGAM
import os as _os

from joblib import dump, load

def createWMLSplot(dfSub, fname):
	allref = pd.read_csv('/refs/WMLS_combinedrefs.csv').dropna()
	allref['Date'] = pd.to_datetime(allref.Date)
	allref = allref.sort_values(by='Date')
	# Get first-time points only
	allref = allref.drop_duplicates(subset=['PTID'], keep='first')
	# Binarize sex
	allref['Sex'].replace(0, 'F',inplace=True)
	allref['Sex'].replace(1, 'M',inplace=True)
	# same sex selection
	allref = allref[allref.Sex == dfSub.loc[0,'Sex']]

	# Get only those reference subjects ±3 years from subject age
	lowlim = int(dfSub['Age'].values[0]) - 3
	uplim = int(dfSub['Age'].values[0]) + 3

	allref = allref.dropna(subset=['604'])

	### Get AD reference values
	ADRef = allref.loc[allref['Diagnosis_nearest_2.0'] == 'AD']

	### Get CN reference values
	CNRef = allref.loc[allref['Diagnosis_nearest_2.0'] == 'CN']

	### Get CN data points to create GAM lines with
	X_CN = CNRef.Age.values.reshape([-1,1])
	y_CN = CNRef['604'].values.reshape([-1,1])

	### Get AD data points to create GAM lines with
	X_AD = ADRef.Age.values.reshape([-1,1])
	y_AD = ADRef['604'].values.reshape([-1,1])

	totCNsubjs = CNRef.shape[0]
	shownCNsubjs = CNRef.loc[(lowlim <= CNRef['Age']) & (CNRef['Age'] <= uplim)].shape[0]

	if dfSub['Sex'].values[0] == 'M':
		sex = 'male'
	else:
		sex = 'female'

	##############################################################
	########### CN values ###########
	# fit the mean model first by CV
	gam50 = ExpectileGAM(expectile=0.5).gridsearch(X_CN, y_CN)

	# and copy the smoothing to the other models
	lam = gam50.lam

	# fit a few more models
	gam90 = ExpectileGAM(expectile=0.90, lam=lam).fit(X_CN, y_CN)
	gam10 = ExpectileGAM(expectile=0.10, lam=lam).fit(X_CN, y_CN)
	
	XX_CN = gam50.generate_X_grid(term=0, n=100)
	XX90_CN = list(gam90.predict(XX_CN).flatten())
	XX50_CN = list(gam50.predict(XX_CN).flatten())
	XX10_CN = list(gam10.predict(XX_CN).flatten())
	XX_CN = list(XX_CN.flatten())

	########### AD values ###########
	# fit the mean model first by CV
	gam50 = ExpectileGAM(expectile=0.5).gridsearch(X_AD, y_AD)

	# and copy the smoothing to the other models
	lam = gam50.lam

	# fit a few more models
	gam90 = ExpectileGAM(expectile=0.90, lam=lam).fit(X_AD, y_AD)
	gam10 = ExpectileGAM(expectile=0.10, lam=lam).fit(X_AD, y_AD)
	
	XX_AD = gam50.generate_X_grid(term=0, n=100)
	XX90_AD = list(gam90.predict(XX_AD).flatten())
	XX50_AD = list(gam50.predict(XX_AD).flatten())
	XX10_AD = list(gam10.predict(XX_AD).flatten())
	XX_AD = list(XX_AD.flatten())

	CN_up5 = [abs(int(i)-(int(dfSub['Age'].values[0])+3)) for i in XX_CN]
	CN_down5 = [abs(int(i)-(int(dfSub['Age'].values[0])-3)) for i in XX_CN]

	setmax = max([XX90_CN[CN_up5.index(min(CN_up5))],XX90_CN[CN_down5.index(min(CN_down5))],CNRef['604'].tolist()[0]])
	setmin = min([XX10_CN[CN_up5.index(min(CN_up5))],XX10_CN[CN_down5.index(min(CN_down5))],CNRef['604'].tolist()[0]])
	spacer = (setmax - setmin)*.2
	setmax = setmax + spacer
	setmin = max(0,setmin - spacer)

	fig = py.subplots.make_subplots(rows=1,cols=1,subplot_titles=['Total White Matter Hyperintensity'],vertical_spacing = 0,horizontal_spacing = 0.05)

	fig.append_trace( go.Scatter(mode='markers', x=CNRef.Age.tolist(), y=CNRef['604'].tolist(), legendgroup='CN', marker=dict(color='MediumBlue', size=5), opacity=0.2, name='CN Reference', showlegend=True),1,1)
	fig.append_trace( go.Scatter(mode='lines', x=XX_CN, y=XX90_CN, legendgroup='CN', marker=dict(color='MediumPurple', size=10), name='90th percentile for CN', showlegend=True),1,1)
	fig.append_trace( go.Scatter(mode='lines', x=XX_CN, y=XX50_CN, legendgroup='CN', marker=dict(color='MediumVioletRed', size=10), name='50th percentile for CN', showlegend=True),1,1)
	fig.append_trace( go.Scatter(mode='lines', x=XX_CN, y=XX10_CN, legendgroup='CN', marker=dict(color='MediumBlue', size=10), name='10th percentile for CN', showlegend=True),1,1)
	fig.append_trace( go.Scatter(mode='markers', x=dfSub.Age.tolist(), y=dfSub.WMLS.tolist(),marker=dict(color='Black', symbol = 'circle-cross-open', size=16,line=dict(color='MediumPurple', width=3)), name='Patient', showlegend=True),1,1)

	fig.add_annotation(xref='x domain',yref='y domain',x=0.01,y=0.95, text='Volume (mL<sup>3</sup>): ' + str(int(dfSub['WMLS'].values[0])), showarrow=False,row=1, col=1)

	fig.update_xaxes(range=[lowlim, uplim])
	fig.update_yaxes(range=[setmin, setmax])
	fig.update_annotations(font_size=25)
	fig['layout']['xaxis']['title']='<b>Age (Years)'
	fig['layout']['yaxis']['title']='<b>ICV-normalized Volume (mL<sup>3</sup>)'

	fig.add_annotation(text='<b>{} {} subjects shown (CN: {}). Percentiles calculated over {} {} subjects (CN: {}).<br>Volumes normalized with respect to the intracranial volume (ICV).'.format(shownCNsubjs, sex, shownCNsubjs, totCNsubjs, sex, totCNsubjs), 
                    align='left',
                    showarrow=False,
                    xref='paper',
                    yref='paper',
                    font=dict(size=18),
                    x=.99,
                    y=-.2,
                    bordercolor='black',
                    borderwidth=1)

	fig.for_each_annotation(lambda a: a.update(text=f'<b>{a.text}</b>'))
	fig.update_xaxes(title_font=dict(size=30, family='Times New Roman'))
	fig.update_yaxes(title_font=dict(size=30, family='Times New Roman'))
	fig.update_layout(title_font_family="Times New Roman",legend=dict(xanchor='left',yanchor='top',font=dict(family='Times New Roman',size=18), orientation='h'),margin_b=0,margin_l=0,margin_r=0,margin_t=34)
	fig.write_image(fname, scale=1, width=1920, height=540)