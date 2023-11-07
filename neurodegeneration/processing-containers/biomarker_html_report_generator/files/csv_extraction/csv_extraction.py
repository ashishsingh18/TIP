import pickle
import os as _os
import pandas as pd
import numpy as np
import csv

def extractTable1(dfRef,dfSub):
	names = ['Brain', 'Ventricle', 'Gray Matter', 'White Matter', 'Brainstem', 'Cerebellum', 'Hippocampus']

	for key in names:
		totalnorm = round((dfSub["Total " + key + " Volume"].values[0] - dfRef["Total " + key + " Volume"].mean())/dfRef["Total " + key + " Volume"].std(),2)

		if 'Brainstem' in key:
			dfSub["Brainstem Bilateral Z-score"] = totalnorm
		# If roi does have L/R component
		else:
			Rnorm = round((dfSub["Right " + key + " Volume"].values[0] - dfRef["Right " + key + " Volume"].mean())/dfRef["Right " + key + " Volume"].std(),2)
			Lnorm = round((dfSub["Left " + key + " Volume"].values[0] - dfRef["Left " + key + " Volume"].mean())/dfRef["Left " + key + " Volume"].std(),2)
			AInorm = round((dfSub[key + " AI"].values[0] - dfRef[key + " AI"].mean())/dfRef[key + " AI"].std(),2)
			
			# Add Whole prefix to Brain ROI
			if key == 'Brain':
				dfSub['Total Brain Bilateral Z-score'] = totalnorm
				dfSub['Total Brain AI Z-score'] = AInorm
				dfSub['Total Brain L Z-score'] = Lnorm
				dfSub['Total Brain R Z-score'] = Rnorm
			else:
				dfSub[key + ' Bilateral Z-score'] = totalnorm
				dfSub[key + ' AI Z-score'] = AInorm
				dfSub[key + ' L Z-score'] = Lnorm
				dfSub[key + ' R Z-score'] = Rnorm

	return dfSub

def extractTable2(dfRef,dfSub,mydict,MUSErois,path):
	maphemi = pd.read_csv('/refs/MUSE_ROI_Dictionary.csv')
	all_entries = ""

	# Identify L and R pairs of single ROIs
	Rinds = list(maphemi.loc[(maphemi['HEMISPHERE'] == 'R'), 'ROI_INDEX'].values)
	Rinds_name = list(maphemi.loc[(maphemi['HEMISPHERE'] == 'R'), 'ROI_NAME'].values)
	Linds = list(maphemi.loc[(maphemi['HEMISPHERE'] == 'L'), 'ROI_INDEX'].values)
	Linds_name = list(maphemi.loc[(maphemi['HEMISPHERE'] == 'L'), 'ROI_NAME'].values)

	R = dict(zip(Rinds_name, Rinds))
	L = dict(zip(Linds_name, Linds))

	AI_zscores = {}
	AI_subj = {}

	# Calculate asymmetry for each ROI-pairing - on both subject and reference z-score levels
	for i in L.keys():
		# Join into one string for right ROI
		base = i.split()
		base[0] = 'Right'
		j = ' '.join(base)

		# Calculate AI column for ROI-pairing
		AI_ref = (dfRef[str(L[i])] - dfRef[str(R[j])]).abs().div((dfRef[str(L[i])] + dfRef[str(R[j])])/2, axis = 0)
		AI = abs(MUSErois[i] - MUSErois[j])/((MUSErois[i] + MUSErois[j])/2)

		# Add to AI_zscore dictionary with format 'Base name of single ROI: AI z-score'
		base[1] = base[1].title()
		basename = ' '.join(base[1:])
		AI_zscores[basename] = abs(round((AI - AI_ref.mean())/AI_ref.std(),2))
		AI_subj[basename] = round(AI,2)

	# Identify flagrant z-scores (anything >=abs(1.28))
	priority_dict = {}
	lowercase = lambda s: s[:1].lower() + s[1:]
	for i in AI_subj.keys():
		left = 'Left ' + lowercase(i)
		right = 'Right ' + lowercase(i)

		priority = max(abs(mydict[left]),abs(mydict[right]),AI_zscores[i])
		priority_dict[i] = priority

	# After getting all abnormal ROI names, rank by extrema and then place into table with R vol, L vol, R-zscore, L-zscore, AI, and Z-zscore
	orderedbases = [k for k, v in sorted(priority_dict.items(), key=lambda item: item[1], reverse=True)]

	for index, key in enumerate(orderedbases):
		tbody = ""
		left = 'Left ' + lowercase(key)
		right = 'Right ' + lowercase(key)
		l_zscore = mydict[left]
		r_zscore = mydict[right]
		l_vol = str(round(MUSErois[left]/1000,1))
		r_vol = str(round(MUSErois[right]/1000,1))
		ai = str(AI_subj[key])
		ai_zscore = AI_zscores[key]
		lobe = list(maphemi.loc[(maphemi['ROI_NAME'] == str(left)), 'SUBGROUP_0'].values)[0]

		dfSub[key + ' - Right Volume'] = r_vol
		dfSub[key + ' - Left Volume'] = l_vol
		dfSub[key + ' - L Z-score'] = round(l_zscore,2)
		dfSub[key + ' - R Z-score'] = round(r_zscore,2)
		dfSub[key + ' - AI'] = ai
		dfSub[key + ' - AI Z-score'] = round(ai_zscore,2)

	dfSub.to_csv(path)

def _main(dfSub, dfRef, WMLSref, allz_num, allz, all_MuseROIs_name, spareAD, spareBA, pdf_path):
	UID = _os.path.basename(pdf_path.removesuffix(".pdf"))
	out = _os.path.dirname(pdf_path)

	alldata = dfSub

	dfSub['SPARE_AD'] = spareAD
	dfSub['SPARE_BA'] = spareBA

	dfSub = extractTable1(dfRef,dfSub)
	extractTable2(dfRef,dfSub,allz,all_MuseROIs_name,_os.path.join(out,UID+'_info.csv'))