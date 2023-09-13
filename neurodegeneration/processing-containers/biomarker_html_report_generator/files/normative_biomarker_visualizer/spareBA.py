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

import plotly.graph_objects as go
from pygam import ExpectileGAM
from pygam.datasets import mcycle
from pygam import LinearGAM
import os as _os

from joblib import dump, load

from .boxoffplot import WMHbox

def calculateSpareBA(age,sex,test):
	MUSE_Ref_Values = pd.read_csv('../refs/combinedharmonized_out.csv')

	MUSE_Ref_Values['Date'] = pd.to_datetime(MUSE_Ref_Values.Date)

	MUSE_Ref_Values = MUSE_Ref_Values.sort_values(by='Date')
	MUSE_Ref_Values_hc = MUSE_Ref_Values.drop_duplicates(subset=['PTID'], keep='first')
	MUSE_Ref_Values_pt = MUSE_Ref_Values[~MUSE_Ref_Values.isin(MUSE_Ref_Values_hc)].dropna()

	MUSE_Ref_Values = MUSE_Ref_Values.drop(['MRID','Study','PTID','SITE','Date', 'SPARE_AD', 'Diagnosis_nearest_2.0'], axis=1, inplace=False)

	# Make dataframe from patient's age, sex, and ROI dictionaries
	test = pd.DataFrame(test, index=[0])
	test.drop('702',axis=1,inplace=True)
	MUSE_Ref_Values.drop('702',axis=1,inplace=True)
	test['Age'] = age
	if sex == "F":
		test['Sex'] = 0
	else:
		test['Sex'] = 1

	# Order test data's columns to those of the reference data
	MUSE_Ref_Values_tmp = MUSE_Ref_Values.drop('SPARE_BA',axis=1,inplace=False)
	test = test[MUSE_Ref_Values_tmp.columns]

	### Make numpy arrays of zeros to store results for the classifier ###
	distances_MUSE_Ref_Values = np.zeros( 1 )
	predictions_MUSE_Ref_Values = np.zeros( 1 )

	### Drop subjects w/o SPARE-BA
	MUSE_Ref_Values = MUSE_Ref_Values.dropna(subset=['SPARE_BA'])

	### Get data to be used in classification analysis ###
	# Training data
	X_train_MUSE_Ref_Values = MUSE_Ref_Values.loc[:,MUSE_Ref_Values.columns != "SPARE_BA"]
	Y_train_MUSE_Ref_Values = MUSE_Ref_Values.loc[:,"SPARE_BA"].values

	#Testing data
	X_test = test.loc[:,test.columns]

	### Actual train + testing ###
	# scale features
	scaler_MUSE_Ref_Values = preprocessing.MinMaxScaler( feature_range=(0,1) ).fit( X_train_MUSE_Ref_Values )
	X_train_MUSE_Ref_Values_norm_sc = scaler_MUSE_Ref_Values.transform( X_train_MUSE_Ref_Values )
	X_test_MUSE_Ref_Values_norm_sc = scaler_MUSE_Ref_Values.transform( X_test )

	# load classifers
	svr_MUSE_Ref_Values = svm.SVR( kernel='linear', C=1 );

	# fit classifers
	svr_MUSE_Ref_Values.fit( X_train_MUSE_Ref_Values_norm_sc, Y_train_MUSE_Ref_Values )

	# Save newly trained svc model
	#dump(svr_MUSE_Ref_Values, '/data/spareBA.joblib')

	# Load trained reference model
	#svr_MUSE_Ref_Values = load('/models/spareBA.joblib') 

	# get distance and predictions for the test subject
	predictions_MUSE_Ref_Values = svr_MUSE_Ref_Values.predict( X_test_MUSE_Ref_Values_norm_sc )

	return predictions_MUSE_Ref_Values

def createSpareBAplot(spareBA, dfSub, dfRef, fig, row, shownCNsubjs, shownADsubjs, shownCNsubjs_WMLS, totCNsubjs, totADsubjs, totCNsubjs_WMLS, fname):
	allref = dfRef

	# Age and sex filter
	#allref = allref[allref.Sex == dfSub['Sex'].values[0]]

	if dfSub['Sex'].values[0] == 'M':
		sex = 'male'
	else:
		sex = 'female'

	lowlim = int(dfSub['Age'].values[0]) - 3
	uplim = int(dfSub['Age'].values[0]) + 3
	allref = allref.dropna(subset=['SPARE_BA'])	

	dfSub['SPARE_BA'] = spareBA

	### Get AD reference values
	ADRef = allref.loc[allref['Diagnosis_nearest_2.0'] == 'AD']

	### Get CN reference values
	CNRef = allref.loc[allref['Diagnosis_nearest_2.0'] == 'CN']

	### Get CN data points to create GAM lines with
	X_CN = CNRef.Age.values.reshape([-1,1])
	y_CN = CNRef['SPARE_BA'].values.reshape([-1,1])

	### Get AD data points to create GAM lines with
	X_AD = ADRef.Age.values.reshape([-1,1])
	y_AD = ADRef['SPARE_BA'].values.reshape([-1,1])

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
	AD_up5 = [abs(int(i)-(int(dfSub['Age'].values[0])+3)) for i in XX_AD]
	AD_down5 = [abs(int(i)-(int(dfSub['Age'].values[0])-3)) for i in XX_AD]
	setmax = max([XX90_CN[CN_up5.index(min(CN_up5))],XX90_CN[CN_down5.index(min(CN_down5))],XX90_AD[AD_up5.index(min(AD_up5))],XX90_AD[AD_down5.index(min(AD_down5))],ADRef['SPARE_BA'].tolist()[0],CNRef['SPARE_BA'].tolist()[0]])
	setmin = min([XX10_CN[CN_up5.index(min(CN_up5))],XX10_CN[CN_down5.index(min(CN_down5))],XX10_AD[AD_up5.index(min(AD_up5))],XX10_AD[AD_down5.index(min(AD_down5))],ADRef['SPARE_BA'].tolist()[0],CNRef['SPARE_BA'].tolist()[0]])
	spacer = (setmax - setmin)*.4
 
	## spareBA score too high to be showed
	setmax = max(setmax, spareBA)
	setmin = min(setmin, spareBA)
 
	setmax = setmax + spacer
	setmin = setmin - spacer

	# Create data for unity line
	XX_unity = np.linspace(lowlim,uplim,100)

	fig.append_trace( go.Scatter(mode='markers', x=ADRef.Age.tolist(), y=ADRef['SPARE_BA'].tolist(), legendgroup='AD', marker=dict(color='MediumVioletRed', size=5), opacity=0.2, name='AD Reference', showlegend=False),row,4)
	fig.append_trace( go.Scatter(mode='markers', x=CNRef.Age.tolist(), y=CNRef['SPARE_BA'].tolist(), legendgroup='CN', marker=dict(color='MediumBlue', size=5), opacity=0.2, name='CN Reference', showlegend=False),row,4)
	fig.append_trace( go.Scatter(mode='lines', x=XX_CN, y=XX90_CN, legendgroup='CN', marker=dict(color='MediumPurple', size=10), name='90th percentile for CN', showlegend=False),row,4)
	fig.append_trace( go.Scatter(mode='lines', x=XX_CN, y=XX50_CN, legendgroup='CN', marker=dict(color='MediumVioletRed', size=10), name='50th percentile for CN', showlegend=False),row,4)
	fig.append_trace( go.Scatter(mode='lines', x=XX_CN, y=XX10_CN, legendgroup='CN', marker=dict(color='MediumBlue', size=10), name='10th percentile for CN', showlegend=False),row,4)
	fig.append_trace( go.Scatter(mode='lines', x=XX_AD, y=XX90_AD, legendgroup='AD', marker=dict(color='MediumPurple', size=10), name='90th percentile for AD', line=dict(dash = 'dash'), showlegend=False),row,4)
	fig.append_trace( go.Scatter(mode='lines', x=XX_AD, y=XX50_AD, legendgroup='AD', marker=dict(color='MediumVioletRed', size=10), name='50th percentile for AD', line=dict(dash = 'dash'), showlegend=False),row,4)
	fig.append_trace( go.Scatter(mode='lines', x=XX_AD, y=XX10_AD, legendgroup='AD', marker=dict(color='MediumBlue', size=10), name='10th percentile for AD', line=dict(dash = 'dash'), showlegend=False),row,4)
	fig.append_trace( go.Scatter(mode='lines', x=XX_unity, y=XX_unity, marker=dict(color='Black', size=10), line=dict(dash = 'dot'), showlegend=False),row,4)
	fig.append_trace( go.Scatter(mode='markers', x=dfSub.Age.tolist(), y=dfSub.SPARE_BA.tolist(),marker=dict(color='Black', symbol = 'circle-cross-open', size=16,line=dict(color='MediumPurple', width=3)), name='Patient', showlegend=False),row,4)

	fig.update_xaxes(range=[lowlim, uplim])
	fig['layout']['yaxis8'].update(range=[setmin, setmax])
	fig['layout']['yaxis8']['title']='<b>Predicted brain age'
	fig['layout']['xaxis8']['title']='<b>Age (Years)'
	fig['layout']['xaxis8']['title'].update(font=dict(size=18))
	fig['layout']['yaxis8']['title'].update(font=dict(size=18))

	score = str(round(spareBA,1))
	gapnum = round(spareBA-dfSub.Age.tolist()[0],1)

	if gapnum > 0:
		gap = '+' + str(gapnum)
	elif gapnum < 0:
		gap = '-' + str(gapnum)
	else:
		gap = str(gapnum)

	fig.add_annotation(xref='x domain',yref='y domain',x=0.01,y=0.95, text='Score: ' + score, showarrow=False,row=row, col=4)
	fig.add_annotation(xref='x domain',yref='y domain',x=0.01,y=0.91, text='Gap: ' + gap, showarrow=False,row=row, col=4)

	fig.add_annotation(text='<b>{} {} subjects with regional volumes shown (CN: {}; AD: {}).<br>Regional volume percentiles calculated over {} {} subjects (CN: {}; AD: {}).<br>{} {} subjects with lesion volume shown (CN: {}).<br>Lesion volume percentiles calculated over {} {} subjects (CN: {}).<br>All volumes normalized with respect to the intracranial volume (ICV).'.format(shownADsubjs+shownCNsubjs, sex, shownCNsubjs, shownADsubjs, totADsubjs+totCNsubjs, sex, totCNsubjs, totADsubjs, shownCNsubjs_WMLS, sex, shownCNsubjs_WMLS, totCNsubjs_WMLS, sex, totCNsubjs_WMLS),
                    align='left',
                    showarrow=False,
                    xref='paper',
                    yref='paper',
                    font=dict(size=20),
                    x=.966,
                    y=-.23,
                    bordercolor='black',
                    borderwidth=1)

	fig.for_each_annotation(lambda a: a.update(text=f'<b>{a.text}</b>'))

	fig.update_layout(plot_bgcolor = 'rgba(243,243,243,1)',title_font_family="Times New Roman",legend=dict(xanchor='left',y=-.09,font=dict(family='Times New Roman',size=22), orientation='h'),margin_b=0,margin_l=0,margin_r=0,margin_t=25)
	fig.write_image(fname, scale=1, width=1920, height=1080)

	# Box off WMH graph
	WMHbox(fname)