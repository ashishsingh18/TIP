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

# TODO: Add PMC to reference values
def calculateSpareAD(age,sex,test):
	MUSE_Ref_Values = pd.read_csv('../refs/combinedharmonized_out.csv')

	MUSE_Ref_Values.drop('SPARE_AD',axis=1,inplace=True)

	MUSE_Ref_Values['Date'] = pd.to_datetime(MUSE_Ref_Values.Date)

	MUSE_Ref_Values = MUSE_Ref_Values.sort_values(by='Date')
	MUSE_Ref_Values_hc = MUSE_Ref_Values.drop_duplicates(subset=['PTID'], keep='first')
	MUSE_Ref_Values_pt = MUSE_Ref_Values[~MUSE_Ref_Values.isin(MUSE_Ref_Values_hc)].dropna()

	MUSE_Ref_Values = MUSE_Ref_Values.drop(['MRID','Study','PTID','SITE','Date', 'SPARE_BA'], axis=1, inplace=False)

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
	MUSE_Ref_Values_tmp = MUSE_Ref_Values.drop('Diagnosis_nearest_2.0',axis=1,inplace=False)
	test = test[MUSE_Ref_Values_tmp.columns]

	MUSE_Ref_Values = MUSE_Ref_Values.loc[(MUSE_Ref_Values['Diagnosis_nearest_2.0'] == "CN") | (MUSE_Ref_Values['Diagnosis_nearest_2.0'] == "AD")]
	MUSE_Ref_Values["Diagnosis_nearest_2.0"].replace({"CN": 0, "AD": 1}, inplace=True)

	### Make numpy arrays of zeros to store results for the classifier ###
	distances_MUSE_Ref_Values = np.zeros( 1 )
	predictions_MUSE_Ref_Values = np.zeros( 1 )

	### Get data to be used in classification analysis ###
	# Training data
	X_train_MUSE_Ref_Values = MUSE_Ref_Values.loc[:,MUSE_Ref_Values.columns != "Diagnosis_nearest_2.0"]
	Y_train_MUSE_Ref_Values = MUSE_Ref_Values.loc[:,"Diagnosis_nearest_2.0"].values

	#Testing data
	X_test = test.loc[:,test.columns]

	### Actual train + testing ###
	# scale features
	scaler_MUSE_Ref_Values = preprocessing.MinMaxScaler( feature_range=(0,1) ).fit( X_train_MUSE_Ref_Values )
	X_train_MUSE_Ref_Values_norm_sc = scaler_MUSE_Ref_Values.transform( X_train_MUSE_Ref_Values )
	X_test_MUSE_Ref_Values_norm_sc = scaler_MUSE_Ref_Values.transform( X_test )

	# initiate classifer
	svc_MUSE_Ref_Values = svm.SVC( probability=True, kernel='linear', C=1 );

	# fit classifer
	svc_MUSE_Ref_Values.fit( X_train_MUSE_Ref_Values_norm_sc, Y_train_MUSE_Ref_Values )

	# get distance and predictions for the test subject
	distances_MUSE_Ref_Values = svc_MUSE_Ref_Values.decision_function( X_test_MUSE_Ref_Values_norm_sc )
	predictions_MUSE_Ref_Values = svc_MUSE_Ref_Values.predict( X_test_MUSE_Ref_Values_norm_sc )

	return distances_MUSE_Ref_Values, predictions_MUSE_Ref_Values

def createSpareADplot(spareAD, dfSub, dfRef, fig, row, fname):
	# Age and sex filter
	#allref = allref[allref.Sex == dfSub['Sex'].values[0]]
	lowlim = int(dfSub['Age'].values[0]) - 3
	uplim = int(dfSub['Age'].values[0]) + 3
	allref = dfRef.dropna(subset=['SPARE_AD'])

	dfSub['SPARE_AD'] = spareAD

	### Get AD reference values
	ADRef = allref.loc[allref['Diagnosis_nearest_2.0'] == 'AD']

	### Get CN reference values
	CNRef = allref.loc[allref['Diagnosis_nearest_2.0'] == 'CN']

	### Get CN data points to create GAM lines with
	X_CN = CNRef.Age.values.reshape([-1,1])
	y_CN = CNRef['SPARE_AD'].values.reshape([-1,1])

	### Get AD data points to create GAM lines with
	X_AD = ADRef.Age.values.reshape([-1,1])
	y_AD = ADRef['SPARE_AD'].values.reshape([-1,1])

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
	setmax = max([XX90_CN[CN_up5.index(min(CN_up5))],XX90_CN[CN_down5.index(min(CN_down5))],XX90_AD[AD_up5.index(min(AD_up5))],XX90_AD[AD_down5.index(min(AD_down5))],ADRef['SPARE_AD'].tolist()[0],CNRef['SPARE_AD'].tolist()[0]])
	setmin = min([XX10_CN[CN_up5.index(min(CN_up5))],XX10_CN[CN_down5.index(min(CN_down5))],XX10_AD[AD_up5.index(min(AD_up5))],XX10_AD[AD_down5.index(min(AD_down5))],ADRef['SPARE_AD'].tolist()[0],CNRef['SPARE_AD'].tolist()[0]])
	
 	## spareAD score too high to be showed
	setmax = max(setmax, spareAD)
	setmin = min(setmin, spareAD)
	
	spacer = (setmax - setmin)*.4
	setmax = setmax + spacer
	setmin = setmin - spacer

	# Create data for unity line
	XX_unity = np.linspace(lowlim,uplim,100)

	fig.append_trace( go.Scatter(mode='markers', x=ADRef.Age.tolist(), y=ADRef['SPARE_AD'].tolist(), legendgroup='AD', marker=dict(color='MediumVioletRed', size=5), opacity=0.2, name='AD Reference', showlegend=False),row,3)
	fig.append_trace( go.Scatter(mode='markers', x=CNRef.Age.tolist(), y=CNRef['SPARE_AD'].tolist(), legendgroup='CN', marker=dict(color='MediumBlue', size=5), opacity=0.2, name='CN Reference', showlegend=False),row,3)
	fig.append_trace( go.Scatter(mode='lines', x=XX_CN, y=XX90_CN, legendgroup='CN', marker=dict(color='MediumPurple', size=10), name='90th percentile for CN', showlegend=False),row,3)
	fig.append_trace( go.Scatter(mode='lines', x=XX_CN, y=XX50_CN, legendgroup='CN', marker=dict(color='MediumVioletRed', size=10), name='50th percentile for CN', showlegend=False),row,3)
	fig.append_trace( go.Scatter(mode='lines', x=XX_CN, y=XX10_CN, legendgroup='CN', marker=dict(color='MediumBlue', size=10), name='10th percentile for CN', showlegend=False),row,3)
	fig.append_trace( go.Scatter(mode='lines', x=XX_AD, y=XX90_AD, legendgroup='AD', marker=dict(color='MediumPurple', size=10), name='90th percentile for AD', line=dict(dash = 'dash'), showlegend=False),row,3)
	fig.append_trace( go.Scatter(mode='lines', x=XX_AD, y=XX50_AD, legendgroup='AD', marker=dict(color='MediumVioletRed', size=10), name='50th percentile for AD', line=dict(dash = 'dash'), showlegend=False),row,3)
	fig.append_trace( go.Scatter(mode='lines', x=XX_AD, y=XX10_AD, legendgroup='AD', marker=dict(color='MediumBlue', size=10), name='10th percentile for AD', line=dict(dash = 'dash'), showlegend=False),row,3)
	fig.append_trace( go.Scatter(mode='lines', x=XX_unity, y=[0]*100, marker=dict(color='Black', size=10), line=dict(dash = 'dot'), showlegend=False),row,3)
	fig.append_trace( go.Scatter(mode='markers', x=dfSub.Age.tolist(), y=dfSub.SPARE_AD.tolist(),marker=dict(color='Black', symbol = 'circle-cross-open', size=16,line=dict(color='MediumPurple', width=3)), name='Patient', showlegend=False),row,3)
	
	fig.update_xaxes(range=[lowlim, uplim])
	fig.add_annotation(xref='x domain',yref='y domain',x=0.01,y=0.95, text='Score: ' + str(round(spareAD,1)), showarrow=False,row=row,col=3)
	fig['layout']['xaxis7']['title']='<b>Age (Years)'
	fig['layout']['yaxis7'].update(range=[setmin, setmax])
	fig['layout']['yaxis7']['title']='<b>Score'
	fig['layout']['xaxis7']['title'].update(font=dict(size=18))
	fig['layout']['yaxis7']['title'].update(font=dict(size=18))

	return fig