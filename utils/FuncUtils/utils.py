import torch
import torch.nn.functional as F
import random
import os
import yaml
import json
import vtk
import numpy as np
import pandas as pd
from vtk.util.numpy_support import vtk_to_numpy
from torch.autograd import Variable
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from multiprocessing import Pool
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
os.environ["PYTHONIOENCODING"] = "UTF-8"

############################################################################################################
#                                           Utility funtcions                                               #
############################################################################################################

def time_calculator(Tinit, Tfinal):
    """
    Function to calculate the time in hours, minutes and seconds

    Args:
        Tinit: Initial time
        Tfinal: Final time

    Returns:
        h: hours
        m: minutes
        s: seconds
    """

    h = (Tfinal - Tinit) // 3600
    if h<1:
        h = 0
    m = int(((Tfinal - Tinit) % 3600) // 60)
    s = int(((Tfinal - Tinit) % 3600) % 60)
    s = int(s)
    return h, m, s

def load_config(config_path):
    """
    Function to load a configuration file

    Args:
        config_path: Path to the configuration file

    Returns:
        config: Configuration dictionary
    """
    config = {}
    file_extension = config_path.split('.')[-1]
    if file_extension == 'yaml':
        with open(config_path, 'r') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
    elif file_extension == 'json':
        with open(config_path, 'r') as file:
            config = json.load(file)
    return config

############################################################################################################
#                                   Functions to manipulate fibers                                         #
############################################################################################################

def normalize_bundle(fibers, BBB):
    """
    Function to normalize a bundle of fibers with their corresponding brain bounding box

    Args:
        fibers: List of fibers
        BBB: Brain bounding box

    Returns:
        fibers: Normalized fibers
    """

    # Get brain bounding box origin
    brains_origin = BBB[::2]

    # Translate the verts to the brain bounding box origin
    for i, fiber in enumerate(fibers):
        fibers[i] -= brains_origin

    # Get the brain bounding box size
    brains_size = [BBB[i+1] - BBB[i] for i in range(0, len(BBB), 2)]

    # Normalize the verts
    for i, fiber in enumerate(fibers):
        fibers[i] /= brains_size

    return fibers

def normalize_batch(V, BBB, C=None, T=None, tolerance=0.1, epsilon=1e-6):
    """
    Function to normalize the vertices and curvature of a batch of fibers with their corresponding brain bounding box

    Args:
        V: Vertices of the fibers
        C: Curvature values of the fibers
        BBB: Brain bounding box [Batch_size, [x_min, x_max, y_min, y_max, z_min, z_max]]
        tolerance: Tolerance factor to expand the bounding box
        epsilon: Small value to avoid division by zero

    Returns:
        V: Normalized vertices
        C: Normalized curvature
    """

    # Calculate the expanded bounding box
    expanded_BBB = BBB.clone()
    expanded_BBB[:, ::2] -= (BBB[:, 1::2] - BBB[:, ::2]) * tolerance
    expanded_BBB[:, 1::2] += (BBB[:, 1::2] - BBB[:, ::2]) * tolerance

    # Get brain bounding box origin
    brains_origin = expanded_BBB[:, ::2]

    # Translate the verts to the brain bounding box origin
    V -= brains_origin.unsqueeze(1)

    # Get the brain bounding box size
    brains_size = expanded_BBB[:, 1::2] - brains_origin

    # Normalize the verts 
    V /= brains_size.unsqueeze(1)

    if C is not None:
        # Normalize the curvature
        min_curvatures = torch.min(C, dim=1, keepdim=True).values
        max_curvatures = torch.max(C, dim=1, keepdim=True).values
        C = (C - min_curvatures) / (max_curvatures - min_curvatures + epsilon)
    
    if T is not None:
        # Normalize the torsion
        min_torsions = torch.min(T, dim=1, keepdim=True).values
        max_torsions = torch.max(T, dim=1, keepdim=True).values
        T = (T - min_torsions) / (max_torsions - min_torsions + epsilon)

    return V, C, T

def add_noise_to_fibers(fibers, noise_range):
    """
    Function to add different levels of Gaussian noise to each fiber

    Args:
        fibers: Vertices of the fibers, shape [batch_size, N, 3]
        noise_range: Range of noise levels
    
    Returns:
        noisy_fibers: Vertices with noise
    """

    batch_size = fibers.shape[0]
    noise_levels = torch.rand(batch_size, 1, 1, device=fibers.device) * noise_range
    noisy_fibers = fibers + noise_levels * torch.randn_like(fibers)
    
    return noisy_fibers

def shear_fibers(fibers, shear_range):
    """
    Function to shear fibers with a different shear transformation for each fiber

    Args:
        fibers: Vertices of the fibers, shape [batch_size, N, 3]
        shear_range: Range of shear
    
    Returns:
        sheared_fibers: Sheared vertices
    """
    batch_size = fibers.shape[0]

    # Randomly shear the fiber
    shears = torch.rand((batch_size, 3), device=fibers.device) * (shear_range[1] - shear_range[0]) + shear_range[0]
    shear_matrices = torch.eye(3, device=fibers.device).unsqueeze(0).repeat(batch_size, 1, 1)
    shear_matrices[:, 0, 1] = shears[:, 0]
    shear_matrices[:, 0, 2] = shears[:, 1]
    shear_matrices[:, 1, 2] = shears[:, 2]
    shear_matrices[:, 1, 0] = shears[:, 0]
    shear_matrices[:, 2, 0] = shears[:, 1]
    shear_matrices[:, 2, 1] = shears[:, 2]

    sheared_fibers = torch.einsum('bij,bkj->bki', shear_matrices, fibers)
    return sheared_fibers

def rotate_fibers(fibers, angle_range):
    """
    Function to rotate fibers with a different rotation for each fiber

    Args:
        fibers: Vertices of the fibers, shape [batch_size, N, 3]
        angle_range: Range of rotation angles in degrees
    
    Returns:
        rotated_fibers: Rotated vertices
    """
    batch_size = fibers.shape[0]
    angles = torch.tensor([random.uniform(angle_range[0], angle_range[1]) for _ in range(batch_size)], dtype=fibers.dtype, device=fibers.device)
    centroid = torch.mean(fibers, dim=2, keepdim=True)
    rotation_matrices = torch.stack([torch.tensor([[torch.cos(angle), -torch.sin(angle), 0],
                                                    [torch.sin(angle), torch.cos(angle), 0],
                                                    [0, 0, 1]], dtype=fibers.dtype, device=fibers.device) for angle in angles])
    centered_fibers = fibers - centroid
    rotated_fibers = torch.einsum('bij,bkj->bik', centered_fibers, rotation_matrices)
    rotated_fibers += centroid
    return rotated_fibers

def scale_fibers(fibers, scale_range):
    """
    Function to scale fibers with a different scale factor for each fiber

    Args:
        fibers: Vertices of the fibers, shape [batch_size, N, 3]
        scale_range: Range of scaling
    
    Returns:
        scaled_fibers: Scaled vertices
    """    
    batch_size = fibers.shape[0]
    scale_factors = torch.rand(batch_size, 1, 1, device=fibers.device) * (scale_range[1] - scale_range[0]) + scale_range[0]
    scaled_fibers = fibers * scale_factors
    return scaled_fibers

def translate_fibers(fibers, translation_range):
    """
    Function to translate fibers with a different translation for each fiber

    Args:
        fibers: Vertices of the fibers, shape [batch_size, N, 3]
        translation_range: Range of translation
    
    Returns:
        translated_fibers: Translated vertices
    """
    batch_size = fibers.shape[0]
    translations = torch.rand((batch_size, 3), device=fibers.device) * (translation_range[1] - translation_range[0]) + translation_range[0]
    translated_fibers = fibers + translations.unsqueeze(1)
    return translated_fibers

def generate_small_fiber(fiber, num_points=128, length_percent=0.1):
    """
    Function to generate a small fiber from a fiber to help with data augmentation

    Args:
        fiber: Fiber to generate the small fiber from
        num_points: Number of points of the small fiber
        length_percent: Length percentage of the small fiber

    Returns:
        new_fiber: New fiber
    """
    batch_size, total_points, _ = fiber.shape

    # Crop a random portion of the fiber
    pt_idx = int(total_points * length_percent)
    random_start_idx = torch.randint(0, total_points - pt_idx, (batch_size,), device=fiber.device)
    indices = (random_start_idx[:, None] + torch.arange(pt_idx, device=fiber.device)[None, :]) % total_points
    temp_fiber = fiber.gather(1, indices.unsqueeze(-1).expand(-1, -1, 3))

    # Resample the fiber
    steps = torch.linspace(0, pt_idx - 1, num_points, device=fiber.device)
    lower_indices = torch.floor(steps).long()
    upper_indices = torch.ceil(steps).long()
    t = (steps - lower_indices.float()).unsqueeze(-1)
    
    lower_values = temp_fiber[:, lower_indices]
    upper_values = temp_fiber[:, upper_indices]
    new_fibers = lower_values + t * (upper_values - lower_values)

    # Randomly translate the fiber
    FBB_min = torch.min(fiber, dim=1)[0]
    FBB_max = torch.max(fiber, dim=1)[0]
    translations = torch.rand((batch_size, 3), device=fiber.device) * (FBB_max - FBB_min) + FBB_min
    new_fibers += 1.3 * translations.unsqueeze(1)

    # Randomly rotate the fiber
    angles = torch.rand((batch_size, 3), device=fiber.device) * 2 * np.pi
    cos_angles = torch.cos(angles)
    sin_angles = torch.sin(angles)

    rotation_matrices_x = torch.stack([
        torch.ones_like(cos_angles[:, 0]), torch.zeros_like(cos_angles[:, 0]), torch.zeros_like(cos_angles[:, 0]),
        torch.zeros_like(cos_angles[:, 0]), cos_angles[:, 0], -sin_angles[:, 0],
        torch.zeros_like(cos_angles[:, 0]), sin_angles[:, 0], cos_angles[:, 0]
    ], dim=-1).reshape(batch_size, 3, 3)

    rotation_matrices_y = torch.stack([
        cos_angles[:, 1], torch.zeros_like(cos_angles[:, 1]), sin_angles[:, 1],
        torch.zeros_like(cos_angles[:, 1]), torch.ones_like(cos_angles[:, 1]), torch.zeros_like(cos_angles[:, 1]),
        -sin_angles[:, 1], torch.zeros_like(cos_angles[:, 1]), cos_angles[:, 1]
    ], dim=-1).reshape(batch_size, 3, 3)

    rotation_matrices_z = torch.stack([
        cos_angles[:, 2], -sin_angles[:, 2], torch.zeros_like(cos_angles[:, 2]),
        sin_angles[:, 2], cos_angles[:, 2], torch.zeros_like(cos_angles[:, 2]),
        torch.zeros_like(cos_angles[:, 2]), torch.zeros_like(cos_angles[:, 2]), torch.ones_like(cos_angles[:, 2])
    ], dim=-1).reshape(batch_size, 3, 3)

    rotation_matrices = rotation_matrices_z @ rotation_matrices_y @ rotation_matrices_x
    new_fibers = torch.einsum('bij,bkj->bki', rotation_matrices, new_fibers)

    # Randomly shear the fiber
    shears = torch.rand((batch_size, 3), device=fiber.device) * 0.2 - 0.1
    shear_matrices = torch.eye(3, device=fiber.device).unsqueeze(0).repeat(batch_size, 1, 1)
    shear_matrices[:, 0, 1] = shears[:, 0]
    shear_matrices[:, 0, 2] = shears[:, 1]
    shear_matrices[:, 1, 2] = shears[:, 2]
    shear_matrices[:, 1, 0] = shears[:, 0]
    shear_matrices[:, 2, 0] = shears[:, 1]
    shear_matrices[:, 2, 1] = shears[:, 2]

    new_fibers = torch.einsum('bij,bkj->bki', shear_matrices, new_fibers)

    return new_fibers

############################################################################################################
#                                   Loss calculation functions                                             #
############################################################################################################


def confidence_loss(pred, confidence, labels, num_classes, lambda_val, beta, eps=1e-6):
    p = F.softmax(pred, dim=1)

    # Assuring there's no numerical instability
    p = torch.clamp(p, 0. + eps, 1. - eps)
    confidence = torch.clamp(confidence, 0. + eps, 1. - eps)

    labels_onehot = Variable(F.one_hot(labels.view(-1), num_classes=num_classes).float()).cuda() # One hot encoding where the label is 1 and the rest is 0
    b = Variable(torch.bernoulli(torch.Tensor(confidence.shape).uniform_(0, 1))).cuda() # Bernoulli distribution to set the confidence to 1 or 0 randomly (no hints)
    conf = confidence * b + (1 - b)
    p_prime = p * conf + (1 - conf) * labels_onehot # Modified probability

    loss_cls = F.nll_loss(torch.log(p_prime), labels) # Cross entropy loss with the modified probability
    loss_confidence = torch.mean(-torch.log(confidence)) # Confidence loss
    loss = loss_cls + lambda_val * loss_confidence # Total loss

    if torch.any(loss_confidence >= beta):
        lambda_val = lambda_val/0.99
    else:
        lambda_val = lambda_val/1.01

    return loss, lambda_val

############################################################################################################
#                                   Functions to manipulate vtk files                                      #
############################################################################################################

def read_vtk_file(file_path):
    """
    Function to read a vtk file

    Args:
        file_path: Path to the vtk file
    
    Returns:
        polydata: Polydata of the vtk file
    """

    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(file_path)
    reader.Update()
    polydata = reader.GetOutput()
    return polydata

def write_vtk_file(polydata, file_path):
    """
    Function to write a vtk file

    Args:
        file_path: Path to the vtk file
        polydata: Polydata of the vtk file
    
    Returns:
        None
    """

    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(file_path)
    writer.SetInputData(polydata)
    writer.Write()

def fill_polydata(fiber_list):
    """
    Function to fill a vtk polydata with fibers

    Args:
        fiber_list: List of fibers

    Returns:
        polydata: Polydata with the fibers
    """

    points = vtk.vtkPoints()
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.Allocate()

    for fiber in fiber_list:
        line = vtk.vtkPolyLine()
        line.GetPointIds().SetNumberOfIds(len(fiber))
        for i, point in enumerate(fiber):
            id = points.InsertNextPoint(point)
            line.GetPointIds().SetId(i, id)
        polydata.InsertNextCell(line.GetCellType(), line.GetPointIds())
    
    return polydata

def get_bounds(subjects_path):
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(subjects_path)
    reader.Update()
    return reader.GetOutput().GetBounds()

def ExtractFiber(polydata, idx_list):
    """
    Function to get fibers from a vtk polydata

    Args:
        polydata: Vtk polydata
        idx_list: list of indexes of the fibers or a single index
    
    Returns:
        verts: Vertices of the fiber
    """

    # Check if idx_list is not a list and convert it to a list if it's a single index
    single_index = False
    if not isinstance(idx_list, (list, tuple, range)):
        idx_list = [idx_list]
        single_index = True

    fibers = []
    for idx in idx_list:
        fiber = polydata.GetCell(idx)
        points = fiber.GetPoints()
        verts = vtk_to_numpy(points.GetData())
        fibers.append(verts)
    
    fibers = np.array(fibers)

    # If a single index was provided, return only that fiber
    if single_index:
        return fibers[0]

    return fibers

############################################################################################################
#                                   Functions for CSV computing                                            #
############################################################################################################

def compute_tract(COUNT, id, tract, vtk_path, npy_path, df, class_label_dict, x_min, x_max, y_min, y_max, z_min, z_max):
    vtk_path = os.path.join(vtk_path, f"{tract}.vtk")
    npy_path = os.path.join(npy_path, f"{tract}.npy")

    num_cells = read_vtk_file(vtk_path).GetNumberOfCells()
    class_ = tract.split('.')[0].replace("_sampled", "")
    label = class_label_dict['label'][class_label_dict['class'].index(class_)]

    temp_df = pd.DataFrame({'count': [COUNT], 'surf': [npy_path], 'class': [class_], 'id': [id], 'label': [label], 'x_min': [x_min], 'x_max': [x_max], 'y_min': [y_min], 'y_max': [y_max], 'z_min': [z_min], 'z_max': [z_max], 'num_cells': [num_cells]
                            }, columns=['count', 'surf', 'class', 'id', 'label', 'x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max', 'num_cells'])
    
    return pd.concat([df, temp_df], ignore_index=True)

def compute_single_subject(vtk_path, npy_path, df, class_label_dict, COUNT=0):
    id = int(vtk_path.split('/')[-1].split('_')[0])
    print(f'\nComputing subject : {id}')
    vtk_files = os.listdir(vtk_path)
    npy_files = os.listdir(npy_path)
    
    vtk_dict = {os.path.splitext(f)[0]: f for f in vtk_files if f.endswith('.vtk')}
    npy_dict = {os.path.splitext(f)[0]: f for f in npy_files if f.endswith('.npy')}

    brain_path = os.path.join('/'.join(vtk_path.split('/')[:-1]), f'{vtk_path.split("/")[-1].split("_")[0]}_brain.vtk')
    print(f"\tComputing brain : {brain_path}")
    x_min, x_max, y_min, y_max, z_min, z_max = get_bounds(brain_path)

    for tract in vtk_dict.keys():
        if tract in npy_dict.keys():
            if tract.replace("_sampled", "") not in class_label_dict['class']:
                continue
            COUNT += 1
            print(f'\tComputing tract {COUNT} : {tract}')
            df = compute_tract(COUNT, id, tract, vtk_path, npy_path, df, class_label_dict, x_min, x_max, y_min, y_max, z_min, z_max)

    return df, COUNT

def compute_multiple_subjects(vtk_path, npy_path, df, class_label_dict):
    COUNT=0
    vtk_subjects = os.listdir(vtk_path)
    npy_subjects = os.listdir(npy_path)

    for id in vtk_subjects:
        if id in npy_subjects:
            subject_vtk_path = os.path.join(vtk_path, id)
            subject_npy_path = os.path.join(npy_path, id)
            df, COUNT = compute_single_subject(subject_vtk_path, subject_npy_path, df, class_label_dict, COUNT=COUNT)

    return df

def compute_brains_csv(vtk_path, df):
    COUNT = 0
    vtk_files = os.listdir(vtk_path)

    vtk_dict = {os.path.splitext(f)[0]: f for f in vtk_files if f.endswith('.vtk')}

    for file in vtk_dict.keys():
        COUNT += 1
        brain_path_vtk = os.path.join(vtk_path, f"{file}.vtk")
        x_min, x_max, y_min, y_max, z_min, z_max = get_bounds(brain_path_vtk)
        num_cells = read_vtk_file(brain_path_vtk).GetNumberOfCells()
        class_ = 'brain'
        label = 0 # Arbitrary label for the brain
        id = int(file.split('.')[0].replace("_brain", "").replace("_sampled", ""))

        temp_df = pd.DataFrame({'count': [COUNT], 'surf': [brain_path_vtk], 'class': [class_], 'id': [id], 'label': [label], 'x_min': [x_min], 'x_max': [x_max], 'y_min': [y_min], 'y_max': [y_max], 'z_min': [z_min], 'z_max': [z_max], 'num_cells': [num_cells]
                                }, columns=['count', 'surf', 'class', 'id', 'label', 'x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max', 'num_cells'])

        df = pd.concat([df, temp_df], ignore_index=True)

    return df

############################################################################################################
#                                   Functions to calculate curvature                                       #
############################################################################################################

def calculate_batch_curvature(batch):
    """
    Function to calculate the curvature of a batch of fibers using PyTorch

    Args:
        batch: Batch of fibers [batch_size, num_points, 3]
    
    Returns:
        curvatures: Curvatures of the fibers [batch_size, num_points]
    """

    first_derivative = torch.gradient(batch, dim=1)[0]
    second_derivative = torch.gradient(first_derivative, dim=1)[0]
    curvature = torch.linalg.norm(torch.cross(first_derivative, second_derivative, dim=2), dim=2) / (torch.linalg.norm(first_derivative, dim=2)**3 + torch.finfo(batch.dtype).eps)

    return curvature

def curvature_for_fiber(fiber):
    """
    Function to calculate the curvature of a fiber

    Args:
        fiber: Fiber [num_points, 3]

    Returns:
        curvature: Curvature of the fiber [num_points]
    """
    first_derivative = np.gradient(fiber, axis=0)
    second_derivative = np.gradient(first_derivative, axis=0)
    curvature = np.linalg.norm(np.cross(first_derivative, second_derivative), axis=1) / (np.linalg.norm(first_derivative, axis=1)**3 + np.finfo(float).eps)
    return curvature

def calculate_curvature(file_path):
    """
    Function to calculate the curvature of fibers in a vtk file or a numpy file

    Args:
        file_path: Path to the vtk file or numpy file

    Returns:
        curvatures: Curvatures of the fibers [num_fibers, num_points]
    """
    if file_path.endswith('.vtk'):
        poly_data = read_vtk_file(file_path)
        fibers = [np.array(poly_data.GetCell(i).GetPoints().GetData()) for i in range(poly_data.GetNumberOfLines())]
    elif file_path.endswith('.npy'):
        fibers = np.load(file_path)
    with Pool() as pool:
        curvatures = pool.map(curvature_for_fiber, fibers)

    return np.array(curvatures)

############################################################################################################
#                                   Functions to calculate tosrion                                         #
############################################################################################################

def calculate_batch_torsion(batch):
    """
    Function to calculate the torsion of a batch of fibers using PyTorch

    Args:
        batch: Batch of fibers [batch_size, num_points, 3]
    
    Returns:
        torsions: Torsions of the fibers [batch_size, num_points]
    """

    first_derivative = torch.gradient(batch, dim=1)[0]
    T_norm = torch.linalg.norm(first_derivative, dim=2)
    T_norm[T_norm == 0] = torch.finfo(batch.dtype).eps
    T = first_derivative / T_norm.unsqueeze(2)

    second_derivative = torch.gradient(first_derivative, dim=1)[0]
    T_prime_norm = torch.linalg.norm(second_derivative, dim=2)
    T_prime_norm[T_prime_norm == 0] = torch.finfo(batch.dtype).eps
    N = second_derivative / T_prime_norm.unsqueeze(2)

    N_prime = torch.gradient(N, dim=1)[0]
    B = torch.cross(T, N)

    numerator = torch.einsum('bij,bij->bi', B, N_prime)
    denominator = torch.linalg.norm(B, dim=2)**2
    denominator[denominator == 0] = torch.finfo(batch.dtype).eps

    torsion = numerator / denominator

    return torsion

def torsion_for_fiber(points):
    """
    Compute the torsion of a 3D curve defined by an array of points.
    
    Parameters:
    points (numpy.ndarray): Array of shape (N, 3) representing N points in 3D space.
    
    Returns:
    numpy.ndarray: Array of torsion values of length N-3.
    """
    # Calculate first derivatives
    T = np.gradient(points, axis=0)
    T_norm = np.linalg.norm(T, axis=1)[:, np.newaxis]
    
    # Handle potential division by zero in T_norm
    T_norm[T_norm == 0] = np.finfo(float).eps
    T = T / T_norm

    # Calculate second derivatives
    T_prime = np.gradient(T, axis=0)
    T_prime_norm = np.linalg.norm(T_prime, axis=1)[:, np.newaxis]
    
    # Handle potential division by zero in T_prime_norm
    T_prime_norm[T_prime_norm == 0] = np.finfo(float).eps
    N = T_prime / T_prime_norm

    # Calculate third derivatives
    N_prime = np.gradient(N, axis=0)
    
    # Calculate the binormal vector
    B = np.cross(T, N)

    # Calculate the numerator of the torsion formula (T, N, N_prime)
    numerator = np.einsum('ij,ij->i', B, N_prime)

    # Calculate the denominator of the torsion formula (|B|^2)
    denominator = np.linalg.norm(B, axis=1) ** 2
    
    # Handle potential division by zero in denominator
    denominator[denominator == 0] = np.finfo(float).eps
    
    # Calculate torsion
    torsion = numerator / denominator
    
    return torsion

def calculate_torsion(file_path):
    if file_path.endswith('.vtk'):
        poly_data = read_vtk_file(file_path)
        fibers = [np.array(poly_data.GetCell(i).GetPoints().GetData()) for i in range(poly_data.GetNumberOfLines())]
    elif file_path.endswith('.npy'):
        fibers = np.load(file_path)
    with Pool() as pool:
        torsions = pool.map(torsion_for_fiber, fibers)

    return np.array(torsions)

############################################################################################################
#                                   Functions to prune fibers                                              #
############################################################################################################

def calculate_gravity_distance(fiber1, fiber2):
    """
    Function to calculate the distance between the gravity centers of two fibers

    Args:
        fiber1: First fiber
        fiber2: Second fiber
    
    Returns:
        distance: Distance between the gravity centers
    """

    gravity1 = np.mean(fiber1, axis=0)
    gravity2 = np.mean(fiber2, axis=0)
    return np.linalg.norm(gravity2 - gravity1)

def calculate_distance_matrix(fibers, method):
    """
    Function to calculate the distance matrix between fibers

    Args:
        fibers: List of fibers
        method: Method to calculate the distance matrix (gravity or density)
    
    Returns:
        distances: Distance matrix
    """

    if method == 'gravity':
        method = calculate_gravity_distance

    num_fibers = len(fibers)
    with Pool() as pool:
        results = pool.starmap(method, [(fibers[i], fibers[j]) for i in range(num_fibers) for j in range(i, num_fibers)])
    distances = np.empty((num_fibers, num_fibers))
    index = 0
    for i in range(num_fibers):
        for j in range(i, num_fibers):
            distances[i, j] = results[index]
            distances[j, i] = distances[i, j]
            index += 1

    return distances

def find_optimal_threshold(distances):
    """
    Function to find the optimal threshold to prune fibers

    Args:
        distances: Distance matrix
    
    Returns:
        best_threshold: Optimal threshold
    """
    mean_distances = np.mean(distances, axis=1)
    best_threshold = np.median(mean_distances)
    return best_threshold

def gravity_pruning(input_file, output_file=None, save=False):
    if type(input_file) == str:
        poly_data = read_vtk_file(input_file)
    else:
        poly_data = input_file
    fibers = [np.array(poly_data.GetCell(i).GetPoints().GetData()) for i in range(poly_data.GetNumberOfLines())]
    distance_matrix = calculate_distance_matrix(fibers, method='gravity')
    threshold = np.percentile(np.mean(distance_matrix), percentage)

    # Prune fibers based on the threshold
    pruned_fibers = []
    for i, fiber in enumerate(fibers):
        if np.mean(distance_matrix[i]) < threshold:
            pruned_fibers.append(fiber)

    pruned_polydata = fill_polydata(pruned_fibers)

    if save:
        # Write the pruned polydata to a new VTK file
        write_vtk_file(output_file, pruned_polydata)
    else:
        return pruned_polydata

def length_pruning(input_file, output_file=None, save=False):
    if type(input_file) == str:
        poly_data = read_vtk_file(input_file)
    else:
        poly_data = input_file

    fibers = [np.array(poly_data.GetCell(i).GetPoints().GetData()) for i in range(poly_data.GetNumberOfLines())]
    lengths = []
    for fiber in fibers:
        lengths.append(np.linalg.norm(fiber[1:] - fiber[:-1], axis=1).sum())
    # Prune short fibers
    pruned_fibers = [fiber for i, fiber in enumerate(fibers) if lengths[i] > np.mean(lengths) - np.std(lengths)]
    # Prune long fibers
    pruned_fibers = [fiber for i, fiber in enumerate(pruned_fibers) if lengths[i] < np.mean(lengths) + np.std(lengths)]
    pruned_polydata = fill_polydata(pruned_fibers)

    if save:
        write_vtk_file(output_file, pruned_polydata)
    else:
        return pruned_polydata

def find_optimal_n_clusters(distances, range_n_clusters=range(3, 10)):
    """
    Function to find the optimal number of clusters for K-means clustering

    Args:
        distances: Distance matrix
        range_n_clusters: Range of number of clusters
    
    Returns:
        best_n_clusters: Optimal number of clusters
    """

    best_score = -1
    best_n_clusters = 0
    for n in range_n_clusters:
        kmeans = KMeans(n_clusters=n, random_state=42, n_init='auto').fit(distances)
        labels = kmeans.labels_
        score = silhouette_score(distances, labels)
        if score > best_score:
            best_score = score
            best_n_clusters = n
    print(f'Best number of clusters: {best_n_clusters}')
    return best_n_clusters

def gravity_clustering(input_file, output_file=None, save=False):
    """
    Function to prune fibers bundles based on a gravity center calculated distance matrix and clustering

    Args:
        input_file: Path to the input vtk file
        output_file: Path to the output vtk file
    
    Returns:
        None
    """
    if type(input_file) == str:
        poly_data = read_vtk_file(input_file)
    else:
        poly_data = input_file
    fibers = [np.array(poly_data.GetCell(i).GetPoints().GetData()) for i in range(poly_data.GetNumberOfLines())]
    distance_matrix = calculate_distance_matrix(fibers, method='gravity')

    pruned_fibers = []
    #K-means clustering
    n_clusters = find_optimal_n_clusters(distance_matrix)
    kmeans = KMeans(n_clusters=2, random_state=42, n_init='auto').fit(distance_matrix)
    clusters = kmeans.labels_.tolist()

    # Prune fibers based on the clustering
    main_cluster = np.argmax(np.bincount(clusters))
    pruned_fibers = [fiber for i, fiber in enumerate(fibers) if clusters[i] == main_cluster]

    pruned_polydata = fill_polydata(pruned_fibers)

    if save:
        # Write the pruned polydata to a new VTK file
        write_vtk_file(output_file, pruned_polydata)
    else:
        return pruned_polydata

def init_visitation_map(BB, voxel_size):
    """
    Function to initiate the visitation map of the fibers in the polydata

    Args:
        polydata: Polydata of the fibers
        voxel_size: Size of the voxel
    
    Returns:
        visitation_map: Visitation map
    """

    x_min, x_max, y_min, y_max, z_min, z_max = BB

    # Calculate the number of voxels in each direction
    num_voxels_x = int((x_max - x_min) / voxel_size)+1
    num_voxels_y = int((y_max - y_min) / voxel_size)+1
    num_voxels_z = int((z_max - z_min) / voxel_size)+1

    # Create a visitation map
    empty_visitation_map = np.zeros((num_voxels_x, num_voxels_y, num_voxels_z))

    return empty_visitation_map

def get_visitation_map(polydata, empty_visitation_map, voxel_size):
    """
    Function to calculate the visitation map of the fibers in the polydata

    Args:
        polydata: Polydata of the fibers
        BB: Bounding box of the polydata
        voxel_size: Size of the voxel
    
    Returns:
        visitation_map: Visitation map
    """

    visitation_map = empty_visitation_map

    x_min, y_min, z_min = polydata.GetBounds()[::2]

    # Fill the visitation map
    fibers = [np.array(polydata.GetCell(i).GetPoints().GetData()) for i in range(polydata.GetNumberOfLines())]
    for fiber in fibers:
        visited_voxels = set()
        for point in fiber:
            x, y, z = point
            i = int((x - x_min) / voxel_size)
            j = int((y - y_min) / voxel_size)
            k = int((z - z_min) / voxel_size)
            voxel = (i, j, k)
            if voxel not in visited_voxels:
                visitation_map[i, j, k] += 1
                visited_voxels.add(voxel)
    return visitation_map

def get_visitation_scores(visitation_map, polydata, voxel_size):
    """
    Function to calculate the visitation scores of the fibers in the polydata

    Args:
        visitation_map: Visitation map
        polydata : Polydata of the fibers
        BB: Bounding box of the polydata
        voxel_size: Size of the voxel
    
    Returns:
        visitation_scores: List of visitation scores
    """
    
    x_min, y_min, z_min = polydata.GetBounds()[::2]

    # Get the visitiations per fiber
    visitation_scores = []
    fibers = [np.array(polydata.GetCell(i).GetPoints().GetData()) for i in range(polydata.GetNumberOfLines())]
    for fiber in fibers:
        visited_voxels = set()
        visitation = 0
        for point in fiber:
            x, y, z = point
            i = int((x - x_min) / voxel_size)
            j = int((y - y_min) / voxel_size)
            k = int((z - z_min) / voxel_size)
            voxel = (i, j, k)
            if voxel not in visited_voxels:
                visitation += visitation_map[i, j, k]
                visited_voxels.add(voxel)
        visitation_scores.append(visitation)

    # Normalize the visitations
    visitation_scores = [x / len(fibers) for x in visitation_scores]

    # Get the average
    for i, fiber in enumerate(fibers):
        visitation_scores[i] = visitation_scores[i] / len(fiber)
    return visitation_scores

def visitation_pruning(input_file, visitation_scores, threshold, output_file=None, save=False):
    if type(input_file) == str:
        polydata = read_vtk_file(input_file)
    else:
        polydata = input_file
    fibers = [np.array(polydata.GetCell(i).GetPoints().GetData()) for i in range(polydata.GetNumberOfLines())]

    # Prune fibers based on the mean and standard deviation
    pruned_fibers = []    
    removed_fibers_idx = []
    for i, fiber in enumerate(fibers):
        if visitation_scores[i] >= threshold:
            pruned_fibers.append(fiber)
        else:
            removed_fibers_idx.append(i)

    pruned_polydata = fill_polydata(pruned_fibers)

    if save:
        # Write the pruned polydata to a new VTK file
        write_vtk_file(output_file, pruned_polydata)
    else:
        return pruned_polydata, removed_fibers_idx

def iterative_visitation_pruning(input_file, empty_visitation_map, voxel_size, std_dev_factor=0.05, output_file=None, save=False, max_iterations=100):
    """
    Function to iteratively prune fibers until all remaining fibers have visitation scores within a certain range.

    Args:
        input_file: Input file or polydata object of the fibers
        empty_visitation_map: Precomputed empty visitation map
        voxel_size: Size of the voxel
        std_dev_factor: Multiplication factor for standard deviation to determine pruning threshold
        output_file: Output file to save the pruned fibers
        save: Boolean flag to save the pruned polydata to a file
        max_iterations: Maximum number of iterations to prevent infinite loop
    
    Returns:
        pruned_polydata: Pruned polydata
    """
    if type(input_file) == str:
        polydata = read_vtk_file(input_file)
    else:
        polydata = input_file

    visitation_map = get_visitation_map(polydata, empty_visitation_map, voxel_size)
    iteration_count = 0

    while iteration_count < max_iterations:
        iteration_count += 1

        visitation_scores = get_visitation_scores(visitation_map, polydata, voxel_size)

        # Calculate mean and standard deviation of visitation scores
        mean_score = np.mean(visitation_scores)
        std_dev = np.std(visitation_scores)

        # Check if the standard deviation is below the threshold
        print(f"Iteration {iteration_count}: Mean score*std_dev_factor: {mean_score*std_dev_factor}, Standard deviation: {std_dev}")
        if std_dev < std_dev_factor * mean_score:
            break

        pruned_polydata, removed_fibers_idx = visitation_pruning(polydata, visitation_scores, mean_score, std_dev * std_dev_factor)

        # Break if no fibers are pruned in this iteration
        if len(removed_fibers_idx) < int(0.05 * len(visitation_scores)):  # Remove 5% of fibers or fewer
            break

        # Update polydata with pruned fibers for the next iteration
        polydata = pruned_polydata

    if save:
        # Write the final pruned polydata to a new VTK file
        write_vtk_file(output_file, polydata)
    else:
        return polydata    

def cluster_and_prune(input_file, visitation_scores, num_clusters=2, output_file=None, save=False):
    if type(input_file) == str:
        polydata = read_vtk_file(input_file)
    else:
        polydata = input_file

    fibers = [np.array(polydata.GetCell(i).GetPoints().GetData()) for i in range(polydata.GetNumberOfLines())]

    # Reshape visitation scores for clustering
    scores = np.array(visitation_scores).reshape(-1, 1)

    # Ensure there are enough unique points for clustering
    unique_scores = np.unique(scores)
    if len(unique_scores) < num_clusters:
        num_clusters = len(unique_scores)

    # Perform K-means clustering
    num_clusters = find_optimal_n_clusters(scores)
    kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=0)
    labels = kmeans.fit_predict(scores)
    cluster_centers = kmeans.cluster_centers_

    min_center = np.argmin(cluster_centers)

    pruned_fibers = []
    removed_fibers_idx = []

    for i, label in enumerate(labels):
        if label != min_center:
            pruned_fibers.append(fibers[i])
        else:
            removed_fibers_idx.append(i)

    pruned_polydata = fill_polydata(pruned_fibers)

    if save:
        write_vtk_file(output_file, pruned_polydata)
    else:
        return pruned_polydata, removed_fibers_idx

def iterative_clustering_pruning(input_file, empty_visitation_map, voxel_size, num_clusters=2, output_file=None, save=False, max_iterations=1):
    if type(input_file) == str:
        polydata = read_vtk_file(input_file)
    else:
        polydata = input_file

    visitation_map = get_visitation_map(polydata, empty_visitation_map, voxel_size)
    iteration_count = 0

    while iteration_count < max_iterations:
        iteration_count += 1

        visitation_scores = get_visitation_scores(visitation_map, polydata, voxel_size)

        # If there are no scores, break the loop
        if len(visitation_scores) == 0:
            break

        pruned_polydata, removed_fibers_idx = cluster_and_prune(polydata, visitation_scores, num_clusters)

        num_fibers = pruned_polydata.GetNumberOfLines()

        if len(removed_fibers_idx) == 0:
            break

        polydata = pruned_polydata

    if save:
        write_vtk_file(output_file, polydata)
    else:
        return polydata

def mean_bounding_box_pruning(input_file, output_file=None, save=False):

    if type(input_file) == str:
        polydata = read_vtk_file(input_file)
    else:
        polydata = input_file

    fibers = [np.array(polydata.GetCell(i).GetPoints().GetData()) for i in range(polydata.GetNumberOfLines())]

    # Calculate the mean bounding box
    minx, maxx, miny, maxy, minz, maxz = [], [], [], [], [], []
    for fiber in fibers:
        minx.append(np.min(fiber[:, 0]))
        maxx.append(np.max(fiber[:, 0]))
        miny.append(np.min(fiber[:, 1]))
        maxy.append(np.max(fiber[:, 1]))
        minz.append(np.min(fiber[:, 2]))
        maxz.append(np.max(fiber[:, 2]))
    mean_BB = [np.mean(minx), np.mean(maxx), np.mean(miny), np.mean(maxy), np.mean(minz), np.mean(maxz)]
    # Add tolerance percentage to the bounding box
    tolerance = 0.4
    mean_BB[0] -= tolerance * (mean_BB[1] - mean_BB[0])
    mean_BB[1] += tolerance * (mean_BB[1] - mean_BB[0])
    mean_BB[2] -= tolerance * (mean_BB[3] - mean_BB[2])
    mean_BB[3] += tolerance * (mean_BB[3] - mean_BB[2])
    mean_BB[4] -= tolerance * (mean_BB[5] - mean_BB[4])
    mean_BB[5] += tolerance * (mean_BB[5] - mean_BB[4])

    pruned_fibers = []
    removed_fibers_idx = []

    for i, fiber in enumerate(fibers):
        if np.min(fiber[:, 0]) > mean_BB[0] and np.max(fiber[:, 0]) < mean_BB[1] and np.min(fiber[:, 1]) > mean_BB[2] and np.max(fiber[:, 1]) < mean_BB[3] and np.min(fiber[:, 2]) > mean_BB[4] and np.max(fiber[:, 2]) < mean_BB[5]:
            pruned_fibers.append(fiber)
        else:
            removed_fibers_idx.append(i)

    pruned_polydata = fill_polydata(pruned_fibers)

    if save:
        write_vtk_file(output_file, pruned_polydata)
    else:
        return pruned_polydata, removed_fibers_idx
    

