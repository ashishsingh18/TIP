import os
from glob import glob

from pathlib import Path
import zipfile
import shutil
from os.path import join, relpath, basename, dirname, exists
from kaapana.operators.KaapanaPythonBaseOperator import KaapanaPythonBaseOperator

from datetime import timedelta

from kaapana.operators.KaapanaBaseOperator import KaapanaBaseOperator, default_registry, default_platform_abbr, default_platform_version
from kaapana.blueprints.kaapana_global_variables import BATCH_NAME, WORKFLOW_DIR


class LocalExtractDataFromArchiveOperator(KaapanaPythonBaseOperator):
  
    processed_count = 0
    def unzip_file(self, zip_path, target_path):
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(target_path)

    def start(self, ds, **kwargs):
        print("Starting moule LocalExtractDataFromArchiveOperator...")
        print("kwargs: ", kwargs)

        print('workflow dir: ', WORKFLOW_DIR)
        print("batch name: ", BATCH_NAME)
        print("operator in dir: ", self.operator_in_dir.operator_out_dir)
        print("operator out dir: ", self.operator_out_dir)
        print("operator in dir type: ", type(self.operator_in_dir))

        workflow_dir = WORKFLOW_DIR
        workflow_dir = workflow_dir if workflow_dir.lower() != "none" else None
        assert workflow_dir is not None

        batch_name = BATCH_NAME
        batch_name = batch_name if batch_name.lower() != "none" else None
        assert batch_name is not None

        operator_in_dir = self.operator_in_dir.operator_out_dir
        operator_in_dir = operator_in_dir if operator_in_dir.lower() != "none" else None
        assert operator_in_dir is not None

        operator_out_dir = self.operator_out_dir
        operator_out_dir = operator_out_dir if operator_out_dir.lower() != "none" else None
        assert operator_out_dir is not None

        # File-extension to search for in the input-dir
        input_file_extension = "*.zip" 

        print("##################################################")
        print("#")
        print("# Starting operator LocalExtractDataFromArchiveOperator:")
        print("#")
        print(f"# workflow_dir:     {workflow_dir}")
        print(f"# batch_name:       {batch_name}")
        print(f"# operator_in_dir:  {operator_in_dir}")
        print(f"# operator_out_dir: {operator_out_dir}")
        print("#")
        print("##################################################")
        print("#")
        print("# Starting processing on BATCH-ELEMENT-level ...")
        print("#")
        print("##################################################")
        print("#")

        # Loop for every batch-element (usually series)
        # batch_folders = sorted([f for f in glob(join("/", workflow_dir, batch_name, "*"))])
        run_dir = os.path.join(WORKFLOW_DIR, kwargs['dag_run'].run_id)
        batch_folders = [f for f in glob(os.path.join(run_dir, BATCH_NAME, '*'))]
        print('batch folders: ', batch_folders)
        for batch_element_dir in batch_folders:
            print("batch element dir: ", batch_element_dir)
            print("#")
            print(f"# Processing batch-element {batch_element_dir}")
            print("#")
            element_input_dir = join(batch_element_dir, operator_in_dir)
            element_output_dir = join(batch_element_dir, operator_out_dir)

            # check if input dir present
            if not exists(element_input_dir):
                print("#")
                print(f"# Input-dir: {element_input_dir} does not exists!")
                print("# -> skipping")
                print("#")
                continue

            # creating output dir
            Path(element_output_dir).mkdir(parents=True, exist_ok=True)

            # creating output dir
            input_files = glob(join(element_input_dir, input_file_extension), recursive=True)
            print(f"# Found {len(input_files)} input-files!")

            # Loop for every input-file found with extension 'input_file_extension'
            for input_file in input_files:

                print(f'input file: {input_file}')

                #extract zip into temp location
                # creating temp dir
                tempfolder = join(element_output_dir,"temp")
                Path(tempfolder).mkdir(parents=True, exist_ok=True)

                #extract
                self.unzip_file(input_file,tempfolder)

                #search for user provided folder
                abs_user_folder_path = ""
                user_folder_found = False
                for rootdir, dirs, files in os.walk(tempfolder):
                    for dir in dirs:
                        if dir == self.user_folder_name:
                            user_folder_found = True
                            abs_user_folder_path = os.path.join(rootdir,dir)
                            break
                    if(user_folder_found):
                        break

                print("absolute user folder path: ", abs_user_folder_path)

                #in user provided folder, search for user provided file extension
                user_file = glob(os.path.join(abs_user_folder_path, self.user_filename_extension), recursive=True)[0]
                print("user file: ", user_file)

                # return this file
                output_file_path = os.path.join(element_output_dir, "{}.nii.gz".format(os.path.basename(batch_element_dir)))
                print("output file path: ", output_file_path)
                shutil.copyfile(user_file,output_file_path)

                #delete temp location
                shutil.rmtree(tempfolder)
                
                self.processed_count += 1

        print("#")
        print("##################################################")
        print("#")
        print("# BATCH-ELEMENT-level processing done.")
        print("#")
        print("##################################################")
        print("#")

        if self.processed_count == 0:
            print("#")
            print("##################################################")
            print("#")
            print("##################  ERROR  #######################")
            print("#")
            print("# ----> NO FILES HAVE BEEN PROCESSED!")
            print("#")
            print("##################################################")
            print("#")
            exit(1)
        else:
            print("#")
            print(f"# ----> {self.processed_count} FILES HAVE BEEN PROCESSED!")
            print("#")
            print("# DONE #")

    def __init__(self,
                 dag,
                 user_folder_name = None,
                 user_file_name = None, #optional and not used at the moment
                 user_filename_extension = "*.nii.gz", # *.csv, etc.
                 execution_timeout=timedelta(minutes=10),
                 **kwargs
                 ):

        self.user_folder_name = user_folder_name if user_folder_name is not None else "NONE"
        self.user_file_name = user_file_name if user_file_name is not None else "NONE"
        self.user_filename_extension = user_filename_extension if user_filename_extension is not None else "NONE"

        super().__init__(
            dag=dag,
            name="localextractdatafromarchive",
            python_callable=self.start,
            execution_timeout=execution_timeout,
            **kwargs
        )
