import time
import vtk
import numpy as np
from multiprocessing import Pool
import os
import argparse
from utils.FuncUtils.utils import time_calculator

def bundle_to_np(bundle_file):
    """
    This function takes as input the bundle file and returns a numpy array of shape (number of cells, NUM_POINTS, 3)
    """
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(bundle_file)
    reader.Update()
    bundle = reader.GetOutput()

    n_fibers_in_bundle = bundle.GetNumberOfCells()
    bundle_np = np.empty((n_fibers_in_bundle, NUM_POINTS, 3))

    for i in range(n_fibers_in_bundle):
        fiber = bundle.GetCell(i)
        points = fiber.GetPoints()
        for j in range(NUM_POINTS):
            bundle_np[i,j,:] = points.GetPoint(j)
    
    return bundle_np
    
def process_bundle(file_path):
    print(f"Processing {file_path}")
    # Read the vtk file
    bundle_np = bundle_to_np(file_path)
    # Save the numpy array as a .npy file
    np.save(f'{output}/{file_path.split("/")[-1].split(".")[0]}.npy', bundle_np)

def process_bundle_2(file_path):
    print(f"Processing {file_path}")
    # Read the vtk file
    bundle_np = bundle_to_np(file_path)
    # Save the numpy array as a .npy file
    if os.path.exists(f'{output}/{file_path.split("/")[-2]}'):
        np.save(f'{output}/{file_path.split("/")[-2]}/{file_path.split("/")[-1].split(".")[0]}.npy', bundle_np)
    else:
        os.makedirs(f'{output}/{file_path.split("/")[-2]}')
        np.save(f'{output}/{file_path.split("/")[-2]}/{file_path.split("/")[-1].split(".")[0]}.npy', bundle_np)


def main(args):
    global output
    global NUM_POINTS
    path = args.path
    output = args.output
    NUM_POINTS = args.num_points
    mode = args.mode

    if not os.path.exists(output):
        os.makedirs(output)

    Tinit = time.time()

    if mode == "single":
        for filename in os.listdir(path):
            if filename.endswith('.vtk'):
                file_path = os.path.join(path, filename)
                process_bundle(file_path)
    elif mode == "multiple":
        file_list = []
        for subject in os.listdir(path):
            if not os.path.isdir(os.path.join(path, subject)):
                continue
            for filename in os.listdir(os.path.join(path, subject)):
                if filename.endswith('.vtk'):
                    file_path = os.path.join(path, subject, filename)
                    file_list.append(file_path)
        pool = Pool(7)
        pool.map(process_bundle_2, file_list)
    else:
        print("Invalid mode")
        return

    h,m,s = time_calculator(Tinit, time.time())
    print(f"\n --- Conversion to numpy finished in {h}h{m}m{s}s --- \n")

def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="Path to the vtk file(s)", required=True)
    parser.add_argument("--output", type=str, help="Path to the folder where the npy files will be saved", required=True)
    parser.add_argument("--num_points", type=int, help="Number of points in each fiber", required=True)
    parser.add_argument("--mode", type=str, help="Mode of operation: 'single' or 'multiple' \n chose 'multiple' if you have multiple subjects", required=True)
    return parser

if __name__ == "__main__":
    main(get_argparser().parse_args())
