from datetime import datetime, timedelta

from kaapana.operators.LocalWorkflowCleanerOperator import LocalWorkflowCleanerOperator
from kaapana.operators.LocalGetInputDataOperator import LocalGetInputDataOperator
from kaapana.operators.LocalGetRefSeriesOperator_modded import LocalGetRefSeriesOperator_modded
from kaapana.operators.LocalTaggingOperator import LocalTaggingOperator
from kaapana.operators.LocalDcm2JsonOperator import LocalDcm2JsonOperator
from airflow.utils.dates import days_ago
from airflow.models import DAG

from airflow.utils.log.logging_mixin import LoggingMixin
from kaapana.operators.DcmConverterOperator import DcmConverterOperator
from kaapana.operators.LocalMinioOperator import LocalMinioOperator
from deepmrseg.DeepMRSegOperator import DeepMRSegOperator
from DLICV.DLICVOperator import DLICVOperator
from kaapana.operators.Itk2DcmSegOperator import Itk2DcmSegOperator
from kaapana.operators.DcmSendOperator import DcmSendOperator
from deepmrsegreport.DeepMRSegReportOperator import DeepMRSegReportOperator
from muse.MuseOperator import MuseOperator
from kaapana.operators.Pdf2DcmOperator import Pdf2DcmOperator
from CombineLabels.CombineLabelsOperator import CombineLabelsOperator
from Seg2RGBDicom.Seg2RGBDicomOperator import Seg2RGBDicomOperator
from html2pdf.html2pdfOperator import html2pdfOperator
from SPARECalculator.spareCalculatorOperator import SpareCalculatorOperator
from ROIQuantifier.roiQuantifierOperator import ROIQuantifierOperator
from NormativeBiomarkerVisualizer.normativeBiomarkerVisualizerOperator import BiomarkerOperator
from BrainVisualize.BrainVisualizeOperator import BrainVisualizeOperator
from CSVExtraction.CSVExtractionOperator import CSVExtractionOperator

log = LoggingMixin().log

args = {
    'ui_visible': True,
    'owner': 'kaapana',
    'start_date': days_ago(0),
    'retries': 0,
    'retry_delay': timedelta(seconds=60)
}

dag = DAG(
    dag_id='combined-neuropipeline-split-report',
    default_args=args,
    concurrency=10,
    max_active_runs=1,
    schedule_interval=None
)

get_input_dicom = LocalGetInputDataOperator(
    dag=dag,
    name='get-input-dicom',
    parallel_downloads=5
)

get_T1 = LocalGetRefSeriesOperator_modded(
    dag=dag,
    input_operator=get_input_dicom,
    name="T1",
    search_policy="study_uid",     
    parallel_downloads=5,     
    dicom_tags=[
        {
             'id': '0008103E', #SeriesDescription
             'value': '*T1*'
         },
         {
             'id': '00080008', #Image Type
             'value': '*PRIMARY*'
         }
    ]
)

get_Flair = LocalGetRefSeriesOperator_modded(
    dag=dag,
    input_operator=get_input_dicom,
    name="Flair",
    search_policy="study_uid",     
    parallel_downloads=5,     
    dicom_tags=[
        {
             'id': '00080008', #Image Type
             'value': '*PRIMARY*'
         },
        {
             'id': '0008103E', #SeriesDescription
             'value': '*FLAIR*'
         }
     ]
)

convert_T1 = DcmConverterOperator(dag=dag, input_operator=get_T1,output_format='nii.gz',task_id="T1_to_nii")
convert_Flair = DcmConverterOperator(dag=dag, input_operator=get_Flair, output_format='nii.gz',task_id="Flair_to_nii")

dlicv = DLICVOperator(dag=dag, priority_weight=10000, input_operator=convert_T1, modeldir="/models/DLICV", batch_size=4,task_id="skull_stripping_dlicv")
wmls = DLICVOperator(dag=dag, priority_weight=1,input_operator=convert_Flair, modeldir="/models/WMLS", batch_size=4,task_id="wmls")
applymask_run_muse = MuseOperator(dag=dag, input_operator=convert_T1,mask_operator=dlicv,batch_size=4,task_id="muse-roi-segmentation")
merge_labels = CombineLabelsOperator(dag=dag,input_operator=applymask_run_muse,task_id="merge-rois")

extract_metadata_T1 = LocalDcm2JsonOperator(dag=dag, input_operator=get_T1, delete_private_tags=True,task_id="GetT1Metadata")
extract_metadata_Flair = LocalDcm2JsonOperator(dag=dag, input_operator=get_Flair, delete_private_tags=True,task_id="GetFlairMetadata")

clean = LocalWorkflowCleanerOperator(dag=dag,clean_workflow_dir=False)

# dmrs_report = DeepMRSegReportOperator(
#     dag=dag,
#     input_operator=get_input_dicom,
#     roi_operator=applymask_run_muse,
#     icv_operator=dlicv,
#     dicom_metadata_json_operator=extract_metadata_T1,
#     task_id="biomarker-extraction")

roi_quant = ROIQuantifierOperator(dag=dag,
    dcm_json_operator=extract_metadata_T1,
    dlicv_operator=dlicv,
    dlmuse_operator=applymask_run_muse,
    wmls_operator=wmls,
    task_id='roi-quantification')

spare_calc = SpareCalculatorOperator(dag=dag,
    input_operator=roi_quant,
    task_id='spare-calculation'
    )

computebiomarkers = BiomarkerOperator(dag=dag,
    quant_operator=roi_quant,
    dlmuse_operator=applymask_run_muse,
    spare_operator=spare_calc,
    task_id='biomarker-computation')

csvextract = CSVExtractionOperator(dag=dag,
    quant_operator=roi_quant,
    dlmuse_operator=applymask_run_muse,
    spare_operator=spare_calc,
    task_id='csv-extract')

visualizebrain = BrainVisualizeOperator(dag=dag,
    quant_operator=roi_quant,
    dlmuse_operator=applymask_run_muse,
    task_id='brain-visualize')

report = html2pdfOperator(dag=dag,
    quant_operator=roi_quant,
    biomarker_operator=computebiomarkers,
    brainvis_operator=visualizebrain,
    task_id='report-generation')

seg2dcm = Seg2RGBDicomOperator(dag=dag,
    mask_operator=merge_labels,
    ref_image=convert_T1,
    dicom_metadata_json=extract_metadata_T1,
    color_csv_path="/models/seg2rgbdicom/roi_color_scheme.csv",
    opacity=0.5,
    series_description="ROI Segmentation Overlay on T1",
    task_id="roi-overlay2dcm")

wmlsoverlay2dcm = Seg2RGBDicomOperator(dag=dag,
    mask_operator=wmls,
    ref_image=convert_Flair,
    dicom_metadata_json=extract_metadata_Flair,
    color_csv_path="/models/seg2rgbdicom/wmls_color_scheme.csv",
    opacity=0.5,
    series_number="902",
    series_description="WMLS Segmentation Overlay on Flair",
    task_id="wmls-overlay2dcm")

#put_to_minio = LocalMinioOperator(dag=dag, action='put', action_operators=[dmrs_report], file_white_tuples=('.pdf'),task_id="put_report_to_minio")
put_to_minio = LocalMinioOperator(dag=dag, action='put', action_operators=[report], file_white_tuples=('.pdf'),task_id="put_report_to_minio")
put_csv_to_minio = LocalMinioOperator(dag=dag, action='put', action_operators=[csvextract], file_white_tuples=('.csv'),task_id="put_csv_to_minio")

put_wmls_to_minio = LocalMinioOperator(dag=dag, action='put', action_operators=[wmls], file_white_tuples=('.nrrd'),task_id="save_wmls_to_minio")

alg_name = "Neuro_Analysis"

pdf2dcm = Pdf2DcmOperator(
    dag=dag,
    input_operator=report,
    dicom_operator=get_T1,
    pdf_title=f"CBICA AI Workbench - Neuro Analysis Report {datetime.now().strftime('%d.%m.%Y %H:%M')}",
    delete_input_on_success=False
)

dcmseg_send_pdf = DcmSendOperator(
    dag=dag,
    input_operator=pdf2dcm,
    task_id="dcm_send_pdf"
    #delete_input_on_success=True
)

seg2dcm_send = DcmSendOperator(dag=dag,input_operator=seg2dcm,task_id="dcm_send_seg2dcm")

wmlsoverlay2dcm_send = DcmSendOperator(dag=dag,input_operator=wmlsoverlay2dcm,task_id="dcm_send_wmlsoverlay2dcm")

dcmSeg_wmls = Itk2DcmSegOperator(dag=dag,
    input_operator=get_Flair,
    segmentation_operator=wmls,
    single_label_seg_info="Brain",
    alg_name="wmls",
    series_description=f'wmls - Brain')

dcm_send_wmls = DcmSendOperator(dag=dag, input_operator=dcmSeg_wmls,task_id="send_wmls_2_dcm")

#old pipeline
#T1
get_input_dicom >> get_T1 >> convert_T1 >> dlicv

get_T1 >> convert_T1 >> seg2dcm

get_T1 >> extract_metadata_T1 >> seg2dcm

extract_metadata_T1 >> roi_quant
dlicv >> roi_quant

dlicv >> applymask_run_muse >> roi_quant >> spare_calc 
applymask_run_muse >> computebiomarkers
spare_calc >> computebiomarkers
roi_quant >> computebiomarkers

roi_quant >> report
computebiomarkers >> report

# visualize brain pipeline
applymask_run_muse >> visualizebrain
roi_quant >> visualizebrain
visualizebrain >> report

applymask_run_muse >> merge_labels >> seg2dcm >> seg2dcm_send >> clean
report >> put_to_minio >> clean
report >> pdf2dcm >> dcmseg_send_pdf >> clean

#wmls pipeline
get_input_dicom >> get_Flair >> convert_Flair >> wmlsoverlay2dcm
get_Flair >> extract_metadata_Flair >> wmlsoverlay2dcm
get_input_dicom >> get_Flair >> convert_Flair >> wmls >> put_wmls_to_minio >> clean
wmls >> roi_quant
wmls >> dcmSeg_wmls >> dcm_send_wmls >> clean
wmls >> wmlsoverlay2dcm >> wmlsoverlay2dcm_send >> clean

# extract csv pipeline
applymask_run_muse >> csvextract
spare_calc >> csvextract
roi_quant >> csvextract
csvextract >> put_csv_to_minio >> clean