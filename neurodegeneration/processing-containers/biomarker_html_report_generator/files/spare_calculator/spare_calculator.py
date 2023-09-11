import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.linear_model import Ridge
from sklearn import metrics
from sklearn.base import clone
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix

import pickle

import os as _os

### Function to train and test model to calculate SPARE-AD score for subject
def calculateSpareAD(age,sex,test):
	# Read in reference values
	MUSE_Ref_Values = pd.read_csv('/refs/combinedharmonized_out.csv')
	MUSE_Ref_Values.drop('SPARE_AD',axis=1,inplace=True)
	# Select first timepoint only
	MUSE_Ref_Values['Date'] = pd.to_datetime(MUSE_Ref_Values.Date)
	MUSE_Ref_Values = MUSE_Ref_Values.sort_values(by='Date')
	MUSE_Ref_Values_hc = MUSE_Ref_Values.drop_duplicates(subset=['PTID'], keep='first')
	# Drop NA values
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

	# load classifers
	svc_MUSE_Ref_Values = svm.SVC( probability=True, kernel='linear', C=1 );

	# fit classifers
	svc_MUSE_Ref_Values.fit( X_train_MUSE_Ref_Values_norm_sc, Y_train_MUSE_Ref_Values )

	# Save newly trained svc model
	#dump(svc_MUSE_Ref_Values, '/data/spareAD.joblib')

	# Load trained reference model
	#svc_MUSE_Ref_Values = load('/models/spareAD.joblib') 

	# get distance and predictions for the test subject
	distances_MUSE_Ref_Values = svc_MUSE_Ref_Values.decision_function( X_test_MUSE_Ref_Values_norm_sc )
	predictions_MUSE_Ref_Values = svc_MUSE_Ref_Values.predict( X_test_MUSE_Ref_Values_norm_sc )

	return distances_MUSE_Ref_Values

def calculateSpareBA(age,sex,test):
	# Read in reference values
	MUSE_Ref_Values = pd.read_csv('/refs/combinedharmonized_out.csv')
	# Select first timepoint only
	MUSE_Ref_Values['Date'] = pd.to_datetime(MUSE_Ref_Values.Date)
	MUSE_Ref_Values = MUSE_Ref_Values.sort_values(by='Date')
	MUSE_Ref_Values_hc = MUSE_Ref_Values.drop_duplicates(subset=['PTID'], keep='first')
	# Drop NA values
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

# Main function that decides order in which functions run in this script
def _main(dfSub,all_MuseROIs_num,out_path):
	UID = _os.path.basename(out_path.removesuffix(".pdf"))
	spareAD = calculateSpareAD(dfSub.loc[0,'Age'],dfSub.loc[0,'Sex'],all_MuseROIs_num)
	spareBA = calculateSpareBA(dfSub.loc[0,'Age'],dfSub.loc[0,'Sex'],all_MuseROIs_num)

	out = _os.path.dirname(out_path)

	# Output spareAD and spareBA scores
	with open(_os.path.join(out,UID+'_spareAD.pkl'), 'wb') as pickle_file:
		pickle.dump(spareAD,pickle_file)
	with open(_os.path.join(out,UID+'_spareBA.pkl'), 'wb') as pickle_file:
		pickle.dump(spareBA,pickle_file)