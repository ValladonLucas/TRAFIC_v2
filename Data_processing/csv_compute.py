import pandas as pd
import vtk
import os
import argparse
import json
import time
import sys
sys.path.append("/work/luvallad/project")
from utils.FuncUtils.utils import time_calculator

def GETBOUNDS(subjects_path):
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(subjects_path)
    reader.Update()
    return reader.GetOutput().GetBounds()

def subject_to_csv(subject_path, output_file, df, class_label_dict):
    COUNT = 0
    print(f'Processing {subject_path}')
    brain_path = os.path.join(subject_path, f'{subject_path.split("/")[-1]}_brain.vtk')

    # Calculate the bounds once outside the loop
    x_min, x_max, y_min, y_max, z_min, z_max = GETBOUNDS(brain_path)

    # Use list comprehension for file processing
    files = [file for file in os.listdir(subject_path) if file.endswith(".vtk")]
    for file in files:
        if file.split('.')[0] not in class_label_dict['class']:
            continue
        print(f'\tProcessing {file}')
        vtp_file_path = os.path.join(subject_path, file)

        # Read the .vtp file
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(vtp_file_path)
        reader.Update()

        # Extract the required information from the .vtk file
        surf = vtp_file_path
        class_ = file.split('.')[0]
        id_ = subject_path.split('/')[-1].split('_')[0]
        label = class_label_dict['label'][class_label_dict['class'].index(class_)]
        num_cells = reader.GetOutput().GetNumberOfCells()

        # Append the information to a temporary DataFrame
        temp_df = pd.DataFrame({'': [COUNT], 'surf': [surf], 'class': [class_], 'id': [id_], 'label': [label],
                                'x_min': [x_min], 'x_max': [x_max], 'y_min': [y_min], 'y_max': [y_max],
                                'z_min': [z_min], 'z_max': [z_max], 'num_cells': [num_cells]})
        
        # Concatenate the temporary DataFrame with the main DataFrame
        df = pd.concat([df, temp_df.dropna()])
        COUNT += 1
    df.to_csv(output_file, index=False)
    print('Done!')

def subjects_to_csv(subjects_path, subjects_path_npy, output_file, df, class_label_dict):
    COUNT = 0
    # Sort the lists before zipping them to get a consistent order
    subject_folders = os.listdir(subjects_path)
    subject_folders = [folder for folder in subject_folders if os.path.isdir(os.path.join(subjects_path, folder))]
    subject_folders.sort()

    subject_npy_folders = os.listdir(subjects_path_npy)
    subject_npy_folders = [folder for folder in subject_npy_folders if os.path.isdir(os.path.join(subjects_path_npy, folder))]
    subject_npy_folders.sort()
    
    for subject, subject_npy in zip(subject_folders, subject_npy_folders):
        if os.path.isdir(os.path.join(subjects_path, subject)) and os.path.isdir(os.path.join(subjects_path_npy, subject_npy)) and os.path.isdir(os.path.join(curvatures_path, curvature)) and os.path.isdir(os.path.join(torsions_path, torsion)):

            subject_path = os.path.join(subjects_path, subject)
            subject_path_npy = os.path.join(subjects_path_npy, subject_npy)
            brain_path = os.path.join(subjects_path, f'{subject}_brain.vtk')

            # Calculate the bounds once outside the loop
            x_min, x_max, y_min, y_max, z_min, z_max = GETBOUNDS(brain_path)

            # Use list comprehension for file processing
            files = [file for file in os.listdir(subject_path) if file.endswith(".vtk")]
            npy_files = [file for file in os.listdir(subject_path_npy) if file.endswith(".npy")]
            files.sort()
            npy_files.sort()
            for file, npy_file in zip(files, npy_files):
                if file.split('.')[0].replace('_sampled','') not in class_label_dict['class']:
                    continue
                print(f'\tProcessing {file}')
                vtk_file_path = os.path.join(subject_path, file)
                npy_file_path = os.path.join(subject_path_npy, npy_file)

                # Read the .vtk file
                reader = vtk.vtkPolyDataReader()
                reader.SetFileName(vtk_file_path)
                reader.Update()

                # Extract the required information from the .vtk file
                surf = npy_file_path
                class_ = vtk_file_path.split('/')[-1].split('.')[0].replace('_sampled','')
                id_ = subject_path.split('/')[-1].replace('_tracts','')
                label = class_label_dict['label'][class_label_dict['class'].index(class_)]
                num_cells = reader.GetOutput().GetNumberOfCells()

                # Append the information to a temporary DataFrame
                temp_df = pd.DataFrame({'': [COUNT], 'surf': [surf], 'class': [class_], 'id': [id_], 'label': [label],
                                        'x_min': [x_min], 'x_max': [x_max], 'y_min': [y_min], 'y_max': [y_max],
                                        'z_min': [z_min], 'z_max': [z_max], 'num_cells': [num_cells]})
                
                # Concatenate the temporary DataFrame with the main DataFrame
                df = pd.concat([df, temp_df.dropna()])
                COUNT += 1

    # Save the DataFrame to a CSV file
    df.to_csv(output_file, index=False)


def brain_to_csv(vtk_folder, npy_folder, output_file, df):
    COUNT = 0
    for vtk_file, npy_file in zip(os.listdir(vtk_folder), os.listdir(npy_folder)):
        if vtk_file.endswith(".vtk") and npy_file.endswith(".npy") and vtk_file.split('.')[0] == npy_file.split('.')[0]:
            print(f'Processing {vtk_file}')
            brain_path = os.path.join(vtk_folder, vtk_file)
            subject_id = brain_path.split('/')[-1].split('_')[0]
            x_min, x_max, y_min, y_max, z_min, z_max = GETBOUNDS(brain_path)
            reader = vtk.vtkPolyDataReader()
            reader.SetFileName(brain_path)
            reader.Update()
            if npy_file != None:
                surf = os.path.join(npy_folder, npy_file)
            else:
                surf = os.path.join(vtk_folder, vtk_file)
            class_ = 'brain'
            id_ = subject_id
            label = 100
            num_cells = reader.GetOutput().GetNumberOfCells()
            temp_df = pd.DataFrame({'': [COUNT], 'surf': [surf], 'class': [class_], 'id': [id_], 'label': [label],
                                    'x_min': [x_min], 'x_max': [x_max], 'y_min': [y_min], 'y_max': [y_max],
                                    'z_min': [z_min], 'z_max': [z_max], 'num_cells': [num_cells]})
            df = pd.concat([df, temp_df.dropna()])
            COUNT += 1
    df.to_csv(output_file, index=False)
    


def main(args):
    Tinit = time.time()
    # Create an empty DataFrame to store the information
    df = pd.DataFrame(columns=['', 'surf', 'class', 'id', 'label', 'x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max', 'num_cells'])
    
    output_file = os.path.join(args.output, args.name)

    if args.mode == 'tracts':

        global n_classes
        classes_file = pd.read_csv(args.classes)
        n_classes = len(classes_file['label'])
        class_label_dict = classes_file.to_dict()

        if args.num_subjects == 'single':
            subject_to_csv(args.vtk_path, output_file, df, class_label_dict)
        else:
            subjects_to_csv(args.vtk_path, args.npy_path, output_file, df, class_label_dict)
    elif args.mode == 'brain':
        brain_to_csv(args.vtk_path, args.npy_path, output_file, df)
    else:
        print('Invalid mode')

    h,m,s = time_calculator(Tinit, time.time())
    print(f'\n--- Saved CSV file in {h}h{m}m{s}s ---\n')

def get_argparse():
  parser = argparse.ArgumentParser(description='Compute CSV file with tracts of different subjects or with whole brain tractography', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument('--vtk_path', type=str, help='Path to the subjects', required=True)
  parser.add_argument('--npy_path', type=str, help='Path to the subjects npy', required=True)
  parser.add_argument('--output', type=str, help='Path to the output csv file', required=True)
  parser.add_argument('--name', type=str, help='Name of the output csv file', default='_output.csv')
  parser.add_argument('--mode', type=str, help='Mode to compute the CSV file (tracts/brain)', required=True)
  parser.add_argument('--num_subjects', type=int, help='Number of subjects to process (single | multiple)', default='single')
  parser.add_argument('--classes', type=str, help='Path to the json file with the classes', required=False)

  return parser

if __name__ == '__main__':

    parser = get_argparse()
    args = parser.parse_args()
    main(args)