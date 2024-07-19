import vtk
import pandas as pd
import numpy as np
import multiprocessing as mp
import os
import argparse
import time
import sys
sys.path.append("/work/luvallad/project")
from utils.FuncUtils.utils import time_calculator

def sampling_bundle(bundle, NEW_NUM_POINTS):
    # bundle reading
    input = vtk.vtkPolyDataReader()
    input.SetFileName(bundle)
    input.Update()
    input = input.GetOutput()

    sampledBundle = vtk.vtkPolyData() # Create a new polydata to store all the sampled fibers
    sampledBundle.Allocate() # Allocate the memory for the sampled bundle

    sampledFiber = vtk.vtkPoints() # Create a new vtkPoints to store the sampled fiber

    # Add the sampled fiber to the bundle
    sampledBundle.SetPoints(sampledFiber)

    ids = vtk.vtkIdList() # Create a new vtkIdList to store the ids of the points of the sampled fiber

    nbFibers = input.GetNumberOfCells() # Get the number of fibers in the bundle

    for i in range(nbFibers):
        nbPointsOnFiber = input.GetCell(i).GetNumberOfPoints() # Get the number of points of the current fiber

        # Compute the step between each point of the sampled fiber
        if NEW_NUM_POINTS > 1:
            EPSILON = 1e-6 # Small value to avoid division by zero
            step = (nbPointsOnFiber - 1) / (NEW_NUM_POINTS - 1 + EPSILON)
        else:
            step = 0

        Points = input.GetCell(i).GetPoints() # Get the points of the current fiber        

        # Calculate the new points of the sampled fiber
        for j in range(NEW_NUM_POINTS):
            t = j*step - int(j*step) # Compute the t parameter for the interpolation

            if int(j*step) < (nbPointsOnFiber-1):
                p0 = Points.GetPoint(int(j*step))
                p1 = Points.GetPoint(int(j*step)+1)

                new_x = (1-t)*p0[0] + t*p1[0]
                new_y = (1-t)*p0[1] + t*p1[1]
                new_z = (1-t)*p0[2] + t*p1[2]

                new_p = [new_x, new_y, new_z]
                new_p = np.array(new_p).astype(np.float32)                
            
            else:
                new_p = Points.GetPoint(int(j*step))
                new_p = np.array(new_p).astype(np.float32)

            id = sampledFiber.InsertNextPoint(new_p) # Insert the new point in the sampled fiber
            ids.InsertNextId(id) # Insert the id of the new point in the vtkIdList)

        # Create a new cell for the sampled fiber
        sampledBundle.InsertNextCell(vtk.VTK_POLY_LINE, ids)

        # Clear the vtkIdList for the next fiber
        ids.Reset()

    return sampledBundle

def write_vtk(polydata, filename):
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)
    writer = vtk.vtkPolyDataWriter()
    writer.SetInputData(polydata)
    writer.SetFileName(filename)
    writer.Write()

def process_csv(row):
    surface, sample_id, sample_class, sample_label = row['surf'], row['id'], row['class'], row['label']
    path_sample = surface
    sampledBundle = sampling_bundle(path_sample, NUM_POINTS)
    write_vtk(sampledBundle, f"{output}/{sample_id}_tracts/{sample_class}_sampled.vtk")

def process_path(subject, path):
    path_sample = os.path.join(path, subject)
    sample_class = subject.split('.')[0]
    sampledBundle = sampling_bundle(path_sample, NUM_POINTS)
    write_vtk(sampledBundle, f"{output}/{sample_class}_sampled.vtk")

def main(args):
    global NUM_POINTS
    global output
    NUM_POINTS = args.num_points
    csv_file = args.csv
    path = args.path
    output = args.output
    if not os.path.exists(output):
        os.makedirs(output)

    Tinit = time.time()
    
    if csv_file is not None:
        df = pd.read_csv(csv_file)
        with mp.Pool() as pool:
            pool.map(process_csv, df.to_dict('records'))
    
    elif path is not None:
        for subject in os.listdir(path):
            if subject.endswith(".vtk"):
                print(f'Processing {subject}')
                process_path(subject, path)

    h,m,s = time_calculator(Tinit, time.time())
    print(f"\n --- Sampling finished in {h}h{m}m{s}s --- \n")

def get_argparse():
    parser = argparse.ArgumentParser(description='Sampling of bundles', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--num_points', type=int, help='Number of points to sample', required=True)

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--csv', type=str, help='CSV file with tracts', default=None)
    group.add_argument('--path', type=str, help='Path to the tracts', default=None)

    parser.add_argument('--output', type=str, help='Output directory', required=True)

    return parser

if __name__ == '__main__':
    main(get_argparse().parse_args())
