from IPython.display import HTML as idhtml
from weasyprint import HTML as wp
import os as _os
import pickle
import pandas as pd
import glob
import datetime as datetime
import pytz

## Write the HTML code for display --- TODO: Move to seperate python file
def writeHtml(plots, tables, flagtable, dftable, outName, brainplot):#, wmlstable):#brainplot, wmlsplots, wmlstable):
	all_plot = ""
	all_brainplot = ""
	all_table = ""
	all_flag = ""

	#Generating HTML for plots and tables
	for plot in plots:
		ind = '''<img src=''' + plot + '''></img>'''
		all_plot = all_plot + ind
	for table in tables:
		ind = '' + table + ''
		all_table = all_table + ind
	for plot in brainplot:
		ind = '''<img src=''' + plot + ''' class="center" style="height:337px; width:792px;"></img>'''
		all_brainplot = all_brainplot + ind
	for table in flagtable:
		ind = '' + table + ''
		all_flag = all_flag + ind
  
	#HTML code for report structure
	html_string_test = '''
	<html>
	<head>
		<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.1/css/bootstrap.min.css">
		<style>
			body { margin:0 0; #background:whitesmoke; }
			.toohigh { background: #ADD8E6 !important; color: white; }
			.toolow { background: #FFB6C1 !important; color: white; }
			.norm { background: white !important; color: white; }
			.ball {
				width: 5px;
				height: 5px;
				border-radius: 50%;
				margin: 1px;
				display: inline-block;
			}
			.cerebellum {background-color: rgb(0, 0, 0) !important;}
			.deep-nuclei {background-color: rgb(104, 52, 153) !important;}
			.deep-white {background-color: rgb(165, 165, 165) !important;}
			.frontal {background-color: rgb(176, 36, 23) !important;}
			.limbic {background-color: rgb(234, 51, 35) !important;}
			.midline {background-color: rgb(126, 171, 86) !important;}
			.occipital {background-color: rgb(223, 131, 68) !important;}
			.parietal {background-color: rgb(245, 194, 66) !important;}
			.temporal {background-color: rgb(80, 113, 190) !important;}
			.ventricle {background-color: rgb(79, 173, 234) !important;}

   
			h6 { margin: 0.5em 0 0.5em 0 ! important; 
				position: relative ! important;
				padding: 5px 15px ! important;
				font-family: 'Times', serif ! important;
				border-style: solid ! important;
				border-radius: 25px ! important;
				font-weight: bold ! important;
				}
		</style>
		<style type="text/css">
			@page { margin: 45px 0px; }
			.col-xs-8 {
  				text-align:center ! important;
  				font-size: 10px ! important;
				}
			.col-xs-4 {
  				text-align:center ! important;
  				font-size: 12px ! important;
				margin-top: 18px !important;
				}
			.col-xs-2 {
				font-size: 10px ! important;
				}
			headtitle{
				line-height: 1em ! important;
				}
  			td {
    			padding: 0 5px;
    			text-align: center;
    			max-width: 100%;
  				}
  			th {
  				text-align: center;
  				max-width: 100%;
  				}
  			.header{
				position: fixed;
				left: 0px;
				right: 0px;
				height: 42px;
				top: -45px;
				bottom: 0px;
				display : flex;
				align-items: center;
				justify-content:space-between;
				grid-template-columns: 1fr auto 1fr;
				background-color: #D3D3D3 ! important;
  			}
  			.footer{
  			position: fixed;
  			left: 0px;
  			right: 0px;
  			bottom: -50px;
  			height: 56px;
  			background-color: #EBEBEB ! important;
  			line-height: .8em ! important;
  			}
  			footnote {
				font-size: 11px ! important;
			}
  			.center {
  			display: block;
  			margin-left: auto;
  			margin-right: auto;
  			}
		</style>
		<style>
			table tr td:first-child{
				text-align: left ! important;
			}
			table thead tr th:first-child{
				text-align: left ! important;
			}
			table {
				border: 2px solid black;
				border-collapse: collapse;
  				table-layout: auto;
    			width: 100%;
    			font-size:80%
				}
			th, td {
				border: 1px solid black;
				}
			tr {
  				border: solid;
  				border-width: 1px 0;
  				max-width: 100%;
				}
		</style>
	</head>
	<body>	
		<div class="header";>
			<div class="col-xs-4">
				<p><b>''' + dftable.loc[0].values[0] + ' | ' + str(int(float(dftable.loc[1].values[0]))) + 'y' + ' ' + dftable.loc[2].values[0] + '''<br>Scan Date: ''' + dftable.loc[3].values[0] + '''</b></p>
			</div>

			<div class="col-xs-8">
				<img src="/logos/cbica.png" alt='Sorry the image is broken' style='height:40px; width:300px'>
			</div>
			<div class="col-xs-8">
				<headtitle><b>Neuroanalysis and Imaging Biomarkers Report</b><br></headtitle>
				<headtitle>[datetime]</headtitle> 
			</div>
		</div>
		<div class="container">
			<h6>Comparision of Volumetry with Normative Harmonized Population Values</h6>
			''' + all_plot + '''
		</div>
		<div class="container">
			<h6>Brain Volumetry and Comparison with Normative Harmonized Population Values</h6>
			''' + all_table + '''
		</div>
		<div class="container">
			<h6>Visualization of regional atrophy </h6>
			''' + all_brainplot + '''
		</div>
		<div class="container">
			<h6>Regions that differ the most from normal</h6>
			''' + all_flag + '''
		</div>
		<div class="container">
			<div class="col-xs-2">
				<div class="ball temporal"></div> Temporal
				<br>
				<div class="ball frontal"></div> Frontal
			</div>
			<div class="col-xs-2">
				<div class="ball midline"></div> Midline
				<br>
				<div class="ball limbic"></div> Limbic
			</div>
			<div class="col-xs-2">
				<div class="ball cerebellum"></div> Cerebellum
				<br>
				<div class="ball occipital"></div> Occipital
			</div>
			<div class="col-xs-2">
				<div class="ball deep-white"></div> Deep White
				<br>
				<div class="ball ventricle"></div> Ventricle
			</div>
			<div class="col-xs-2">
				<div class="ball deep-nuclei"></div> Deep Nuclei
				<br>
				<div class="ball parietal"></div> Parietal
			</div>
		</div>
		<br>
		<div class="footer">
			<footnote>Note: These measurements may be useful as an adjunct to other diagnostic evaluations and the broader clinical context. Measurements do not diagnose a specific underlying disease in isolation. Please confirm the quality and applicability of the segmentation(s) prior to considering these values. Asymmetry Index = (|L - R|)/(.5*(L+R)), where L is the left ROI and R is the right ROI.</footnote>
		</div>
	</body>
	</html>'''
 
	now = datetime.datetime.now(pytz.timezone('US/Eastern'))
	formatted_datetime = 'Report Creation Date (US/Eastern): ' + now.strftime("%m/%d/%Y") + ' Time: ' + now.strftime("%H:%M:%S")
	html_string_test = html_string_test.replace("[datetime]", formatted_datetime)
	f = open(outName,'w')
	f.write(html_string_test)
	f.close()

def _main(pdf_path,in_path_biomarker,in_path_quant,in_path_brainvis):
	UID = str(_os.path.basename(pdf_path.removesuffix(".pdf")))
	pdf_out = str(_os.path.dirname(pdf_path))
	brainplot = []
	plots = []
	tables = []
	flagtable = []
	print('Path to Brain Vis')
	print(_os.path.join(in_path_brainvis,UID+'_finalvis.png'))
	brainplot.append(_os.path.join(in_path_brainvis,UID+'_finalvis.png'))
	plots.append(_os.path.join(in_path_biomarker,UID+'_plot.png'))
 
	tables.append(pd.read_pickle(_os.path.join(in_path_biomarker,UID+'_roisubsettable.pkl')))
	flagtable.append(pd.read_pickle(_os.path.join(in_path_biomarker,UID+'_flagtable.pkl')))
	dfPat = pd.read_pickle(_os.path.join(in_path_quant,UID+'_dfPat.pkl'))

	html_out = _os.path.join(str(pdf_out),'report.html')
	writeHtml(plots, tables, flagtable, dfPat, html_out, brainplot)
	print('\nTemp html file created!')
	wp(html_out).write_pdf(pdf_path)
	print('\nPDF plot file created!')


# if __name__ == '__main__':
# 	pdf_path = '/home/diwu/Desktop/F2/2.16.840.1.114362.1.12066432.24920037488.604832326.447.1607.pdf'
# 	in_path_biomarker = '/home/diwu/Desktop/F2/2.16.840.1.114362.1.12066432.24920037488.604832326.447.1607/biomarker-computation'
# 	in_path_quant = '/home/diwu/Desktop/F2/2.16.840.1.114362.1.12066432.24920037488.604832326.447.1607/roi-quantification'
# 	in_path_brainvis = '/home/diwu/Desktop/F2/2.16.840.1.114362.1.12066432.24920037488.604832326.447.1607/brain-visualize'
# 	_main(pdf_path,in_path_biomarker,in_path_quant,in_path_brainvis)
