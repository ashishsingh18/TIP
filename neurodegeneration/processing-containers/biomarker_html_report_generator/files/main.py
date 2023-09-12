
import roi_quantification as rq
import spare_calculation as sc
import biomarker_computation as bc
import do_brain_visualization as bv
import html_creator as hc

def main(dlicv_mask, muse_mask, dcm_json, wmls_mask):
    #Each step is its own separte .py file
    a,b,c,d = do_roi_quantification() #write_output to temp location inside docker container or do not write and directly read the output 

    do_spare_calculation()
    
    do_biomarker_computation()

    do_brain_visualization()

    create_html()

    #output is a single html file + all files required by csv_extraction container