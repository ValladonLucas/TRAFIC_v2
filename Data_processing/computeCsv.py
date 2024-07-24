import pandas as pd
import os
import argparse
import time
from utils.utils import time_calculator, compute_single_subject, compute_multiple_subjects, compute_brains_csv

def main(args):
    Tinit = time.time()
    df = pd.DataFrame({'count': [], 'surf': [], 'class': [], 'id': [], 'label': [], 'x_min': [], 'x_max': [], 'y_min': [], 'y_max': [], 'z_min': [], 'z_max': [], 'num_cells': []},
                      columns=['count', 'surf', 'class', 'id', 'label', 'x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max', 'num_cells'])
    df = df.astype({'count': 'int32', 'surf': 'str', 'class': 'str', 'id': 'int32', 'label': 'int32', 'x_min': 'float32', 'x_max': 'float32', 'y_min': 'float32', 'y_max': 'float32', 'z_min': 'float32', 'z_max': 'float32', 'num_cells': 'int32'})

    output = os.path.join(args.output, args.name)
    if args.mode == 'tracts':

        class_label_dict = pd.read_csv(args.classes).to_dict('list')
        global n_classes
        n_classes = len(class_label_dict['label'])
        
        if args.num_subjects == 'single':
            # Folder that contains vtk and npy folders has to contain the brain vtk file
            df, _ = compute_single_subject(args.vtk_path, args.npy_path, df, class_label_dict)
        elif args.num_subjects == 'multiple':
            df = compute_multiple_subjects(args.vtk_path, args.npy_path, df, class_label_dict)
    elif args.mode == 'brain':
        df = compute_brains_csv(args.vtk_path, df)

    df.to_csv(output, index=False)
    h,m,s = time_calculator(Tinit, time.time())
    print(f"CSV computation time: {h}h {m}m {s}s")


def get_argparse():
    parser = argparse.ArgumentParser(description='Compute CSV file with tracts of different subjects or with whole brain tractography', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--mode', type=str, help='Mode to compute the CSV file (tracts/brain)', required=True)
    parser.add_argument('--num_subjects', type=str, help='Number of subjects to process (single/multiple)', required=True)
    parser.add_argument('--vtk_path', type=str, help='Path to the subjects vtk file', required=True)
    parser.add_argument('--npy_path', type=str, help='Path to the subjects npy file, specify it only if you are processing tracts', default=None)
    parser.add_argument('--output', type=str, help='Path to the output csv file', required=True)
    parser.add_argument('--name', type=str, help='Name of the output csv file', default='_output.csv')
    parser.add_argument('--classes', type=str, help='Path to the json file with the classes names and labels, specify it only if you are processing tracts', default=None)
    
    return parser

if __name__ == '__main__':

    parser = get_argparse()
    args = parser.parse_args()
    main(args)