import pandas as pd
import numpy as np
import spare_scores as spare
import pickle
import os as _os

# Main function that decides order in which functions run in this script
def spare_main(dfSub,all_MuseROIs_num,out_path):
	UID = _os.path.basename(out_path.removesuffix(".pdf"))
 
	df = pd.DataFrame(columns=['Age','Sex','MRID'])
	df.loc[0,['Age','Sex','MRID']] = [dfSub.loc[0,'Age'], dfSub.loc[0,'Sex'], dfSub.loc[0, 'MRID']]
	col_names = ['H_MUSE_Volume_' + i for i in all_MuseROIs_num if int(i) <= 207]
	col_values = [all_MuseROIs_num[i] for i in all_MuseROIs_num if int(i) <= 207]
	df[col_names] = [col_values]
 
	if _os.path.exists('../tmp/SPARE_AD.csv'):
		_os.remove('../tmp/SPARE_AD.csv')
 
	spare_AD = spare.spare_test(df = df,
                                mdl_path    = '/refs/kaapana_spareAD.pkl.gz',
                                key_var     = 'MRID',
                                output      = '../tmp/SPARE_AD.csv',
                                spare_var   = 'SPARE_AD')
	
	spareAD = spare_AD['data']['SPARE_AD'][0]
 
	if _os.path.exists('../tmp/SPARE_BA.csv'):
		_os.remove('../tmp/SPARE_BA.csv')
 
	spare_BA = spare.spare_test(df = df,
                                mdl_path    = '/refs/kaapana_spareBA_cpu.pkl.gz',
                                key_var     = 'MRID',
                                output      = '../tmp/SPARE_BA.csv',
                                spare_var   = 'SPARE_BA')
	
	spareBA = spare_BA['data']['SPARE_BA'][0]

	out = _os.path.dirname(out_path)

	# Output spareAD and spareBA scores
	with open(_os.path.join(out,UID+'_spareAD.pkl'), 'wb') as pickle_file:
		pickle.dump(spareAD,pickle_file)
	with open(_os.path.join(out,UID+'_spareBA.pkl'), 'wb') as pickle_file:
		pickle.dump(spareBA,pickle_file)
  
	return spareAD, spareBA