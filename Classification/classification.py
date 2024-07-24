import os
import vtk
import torch
import numpy as np
import pandas as pd
from vtk.util.numpy_support import vtk_to_numpy
from utils.utils import time_calculator, calculate_batch_curvature, calculate_batch_torsion, normalize_batch, read_vtk_file, init_visitation_map, write_vtk_file, get_visitation_map, get_visitation_scores, visitation_pruning, iterative_clustering_pruning, mean_bounding_box_pruning, length_pruning
from Dataloaders.classification_dataloader import DataModule
import time
import argparse

from models.pointnet import PN
from models.dec import DEC
from models.seqdec import seqDEC
from models.TractCurvNet import TractCurvNet

torch.multiprocessing.set_sharing_strategy('file_system')


def fill_polydata(polydata_list, verts, prediction, confidence=None):
    # Get the polydata object corresponding to the prediction index
    polydata = polydata_list[prediction]
    points = polydata.GetPoints()
    if points is None:
        points = vtk.vtkPoints()
    
    ids = vtk.vtkIdList()

    # Create or retrieve the arrays for confidence and attention
    if confidence is not None:
        if not polydata.GetCellData().HasArray("Confidence"):
            confidence_array = vtk.vtkFloatArray()
            confidence_array.SetName("Confidence")
            polydata.GetCellData().AddArray(confidence_array)
        else:
            confidence_array = polydata.GetCellData().GetArray("Confidence")

    # Add new points to the vtkPoints object and keep track of their ids
    for i, point in enumerate(verts):
        id = points.InsertNextPoint(point)
        ids.InsertNextId(id)

    # Update the points in the polydata
    polydata.SetPoints(points)
    
    # Create a new line (polyline) with the added points
    polydata.InsertNextCell(vtk.VTK_POLY_LINE, ids)

    # Add the confidence value if provided
    if confidence is not None:
        confidence_array.InsertNextValue(confidence)

    # Update the polydata object in the list
    polydata_list[prediction] = polydata

def main(args):
    Tinit = time.time()

    classes = pd.read_csv(args.classes)

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    # Initialize the model
    elif args.model == "PN":
        model = PN
    elif args.model == "DEC":
        model = DEC
    elif args.model == "seqDEC":
        model = seqDEC
    elif args.model == "TractCurvNet":
        model = TractCurvNet
    else:
        print("Invalid model")
        return

    confidence_thresholds = classes['confidence_thresholds']

    with torch.no_grad():
        model = model.load_from_checkpoint(
        checkpoint_path=args.checkpoint_path,
        n_classes=args.num_classes,
        input_size=4,
        k=args.k,
        dropout=True,
        )
        model.eval()
        model.cuda()
        model.freeze()
        print("Model loaded")

        brain_data = DataModule(class_path=args.csv_path,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers)
        brain_data.setup()
        print("Data loaded")

        predictions_array = []
        valid_idx = []
        valid_ids = []
        confidences_array = []
        
        for batch in brain_data.classification_dataloader():
            V, BBB, SID, IDX = (b.cuda() for b in batch)

            C = calculate_batch_curvature(V)
            T = calculate_batch_torsion(V)
            V, C, T = normalize_batch(V, BBB, C, T, 0.1)
            data = torch.cat([V, C.unsqueeze(-1)], dim=2)
            outputs, conf, _ = model(data)

            conf_means = torch.mean(conf, dim=1, keepdim=True)
            predictions = torch.argmax(outputs, dim=1)
            thresholds_tensor = torch.tensor(confidence_thresholds).cuda()

            mask = conf_means.squeeze(1) >= thresholds_tensor[predictions]
            valid_idx.append(IDX[mask])
            valid_ids.append(SID[mask])
            predictions_array.append(predictions[mask])
            confidences_array.append(conf[mask])

    all_idx_array = np.concatenate([idx.cpu().numpy() for idx in valid_idx])
    ids_array = np.concatenate([ids.cpu().numpy() for ids in valid_ids])
    predictions_array = np.concatenate([pred.cpu().numpy() for pred in predictions_array])
    confidences_array = np.concatenate([conf.cpu().numpy() for conf in confidences_array])

    unique_ids = np.unique(ids_array)

    for id in unique_ids: # Iterate over the brains

        # Initialize the polydata list for each class
        polydata_list = []
        sampled_polydata_list = []
        for _ in range(args.num_classes):
            polydata = vtk.vtkPolyData()
            polydata.Allocate()
            points = vtk.vtkPoints()
            polydata.SetPoints(points)
            polydata_list.append(polydata)

            polydata_sampled = vtk.vtkPolyData()
            polydata_sampled.Allocate()
            points_sampled = vtk.vtkPoints()
            polydata_sampled.SetPoints(points_sampled)
            sampled_polydata_list.append(polydata_sampled)

        idx_array = all_idx_array[ids_array == id] # Get the indexes of the fibers of the same brain

        df = pd.read_csv(args.original_brain)
        df.set_index('id', inplace=True)
        original_brain = read_vtk_file(df.loc[id, 'original_path']) # Load the original brain to get real predictions
        sampled_brain = read_vtk_file(df.loc[id, 'sampled_path']) # Load the sampled brain to store confidence and attention scores

        # Voxel size calculation
        brain_bounding_box = original_brain.GetBounds()
        x_shape = brain_bounding_box[1] - brain_bounding_box[0]
        y_shape = brain_bounding_box[3] - brain_bounding_box[2]
        z_shape = brain_bounding_box[5] - brain_bounding_box[4]
        voxel_size = np.mean([x_shape, y_shape, z_shape])/128
        empty_visitation_map = init_visitation_map(brain_bounding_box, voxel_size)

        # Saves the vertices, confidence and attention scores
        attention_list = [[] for _ in range(args.num_classes)]
        for i in range(idx_array.shape[0]): # Iterate over the fibers
            fiber = original_brain.GetCell(idx_array[i]) # Get the fiber from the original brain
            points = fiber.GetPoints() # Get the points of the fiber
            verts = vtk_to_numpy(points.GetData()) # Get the vertices of the fiber
            fill_polydata(polydata_list, verts, predictions_array[i], confidence=confidences_array[i]) # Fill the polydata with the verts and the predicted label

            sampled_fiber = sampled_brain.GetCell(idx_array[i]) # Get the fiber from the sampled brain
            sampled_points = sampled_fiber.GetPoints() # Get the points of the fiber
            sampled_verts = vtk_to_numpy(sampled_points.GetData()) # Get the vertices of the fiber
            fill_polydata(sampled_polydata_list, sampled_verts, predictions_array[i]) # Fill the polydata with the verts and the predicted label


        count = 0
        for i, polydata in enumerate(polydata_list): # Save the bundles into vtk files
            if polydata.GetNumberOfCells() > 10 and i != len(polydata_list)-1:
                bundle_name = classes["class"][i]
                vtk_file = os.path.join(os.path.join(args.save_path, f"{id}_uncleaned"), bundle_name + ".vtk")
                if not os.path.exists(os.path.join(args.save_path, f"{id}_uncleaned")):
                    os.mkdir(os.path.join(args.save_path, f"{id}_uncleaned"))
                write_vtk_file(polydata, vtk_file)
                if bundle_name not in ["Intra-CBLM-I&P_left", 
                                       "Intra-CBLM-I&P_right", 
                                       "Intra-CBLM-PaT_left", 
                                       "Intra-CBLM-PaT_right",
                                       "CPC", 
                                       "CorticoSpinal-Right", 
                                       "CorticoSpinal-Left"]:

                    visitation_map = get_visitation_map(polydata, empty_visitation_map, voxel_size)
                    visitation_scores = get_visitation_scores(visitation_map, polydata, voxel_size)
                    threshold = np.mean(visitation_scores) - np.std(visitation_scores)
                    polydata, _ = visitation_pruning(polydata, visitation_scores, threshold)                

                    polydata = iterative_clustering_pruning(polydata, empty_visitation_map, voxel_size)

                    visitation_map = get_visitation_map(polydata, empty_visitation_map, voxel_size)
                    visitation_scores = get_visitation_scores(visitation_map, polydata, voxel_size)
                    threshold = np.mean(visitation_scores) - np.std(visitation_scores)
                    polydata, _ = visitation_pruning(polydata, visitation_scores, threshold)


                    polydata, _ = mean_bounding_box_pruning(polydata)
                    polydata = length_pruning(polydata)  

                vtk_file = os.path.join(os.path.join(args.save_path, f"{id}"), bundle_name + ".vtk")
                if not os.path.exists(os.path.join(args.save_path, f"{id}")):
                    os.mkdir(os.path.join(args.save_path, f"{id}"))
                write_vtk_file(polydata, vtk_file)
                print(f"Saved {bundle_name}")
                count += 1

        for i, polydata_sampled in enumerate(sampled_polydata_list):
            if polydata_sampled.GetNumberOfCells() > 10:

                # visitation_map = get_visitation_map(polydata_sampled, empty_visitation_map, voxel_size)
                # visitation_scores = get_visitation_scores(visitation_map, polydata_sampled, voxel_size)
                # polydata_sampled, removed_idx = visitation_pruning(polydata_sampled, visitation_scores, 5)

                # visitation_map = get_visitation_map(polydata_sampled, empty_visitation_map,voxel_size)
                # visitation_scores = get_visitation_scores(visitation_map, polydata_sampled, voxel_size)
                # polydata_sampled, removed_idx = visitation_pruning(polydata_sampled, visitation_scores, 80)

                attention_list[i] = np.array(attention_list[i])
                # attention_list[i] = np.delete(attention_list[i], removed_idx, axis=0)

                bundle_name = classes["class"][i]
                vtk_file = os.path.join(os.path.join(args.save_path, f"{id}_sampled"), bundle_name + "_sampled.vtk")
                if not os.path.exists(os.path.join(args.save_path, f"{id}_sampled")):
                    os.mkdir(os.path.join(args.save_path, f"{id}_sampled"))
                # write_vtk_file(polydata_sampled, vtk_file)
                # print(f"Saved {bundle_name}_sampled")


        print(f"\nSaved {count} bundles for brain {id}\n")

    h,m,s = time_calculator(Tinit, time.time())
    print(f"Classification time: {h}h {m}m {s}s")
    
    
def get_argparse():
    parser = argparse.ArgumentParser(description='Trainer for models', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--csv_path', type=str, help='Path to the csv file', required=True)
    parser.add_argument('--original_brain', type=str, help='Csv file wit paths to the original brains', required=True)
    parser.add_argument('--batch_size', type=int, help='Batch size', required=True)
    parser.add_argument('--num_workers', type=int, help='Number of workers', required=True)
    parser.add_argument('--num_points', type=int, help='Number of points', required=True)
    parser.add_argument('--num_classes', type=int, help='Number of classes', required=True)

    parser.add_argument('--model', type=str, help='Model to train', required=True)
    parser.add_argument('--checkpoint_path', type=str, help='Path to the checkpoint', required=True)

    parser.add_argument('--save_path', type=str, help='Path to save the predicted bundles', required=True)

    parser.add_argument('--classes', type=str, help='Path to the csv file with the classes', required=True)
    parser.add_argument('--k', type=int, help='Number of clusters', required=True)

    return parser

if __name__ == "__main__":
    args = get_argparse().parse_args()
    main(args)
