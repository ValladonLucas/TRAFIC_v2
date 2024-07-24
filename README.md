# Automatic Tract Classification Based On Raw Points and Local Curvature

This project has for aim to replace the [TRAFIC](https://github.com/NIRALUser/Trafic.git) model in [AutoTract](https://github.com/NIRALUser/AutoTract.git). The need for replacement arises from the manual cleaning required after classification, which we aim to eliminate.\
This project has been coded in [Python](https://www.python.org/) and models were coded using [PyTorch](https://github.com/pytorch/pytorch.git),[Pytorch-Lightning](https://github.com/Lightning-AI/pytorch-lightning.git) and [MONAI](https://github.com/Project-MONAI/MONAI.git).


## Prerequisites

To run this project, you need a python envrionment. We've made the choice to use [CONDA](https://github.com/conda/conda.git) and particularly [Miniconda](https://docs.anaconda.com/free/miniconda/) to bootstrap a minimal installation that only includes conda and its dependencies.\
To install miniconda on linux, follow the next instructions :
```bash
mkdir ~/miniconda3
cd ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh -u
```
When miniconda is installed, if when you type `conda -h` you have an error, initiate conda :
```bash
~/miniconda3/bin/conda init $SHELL
```
To help with the use of this project we've built an `environment.yml` file with all the necessary dependencies, you just have to run :
```bash
conda env create -f environment.yml
```
## Getting started : The Data

### Resampling the fibers

To train the models, we use [VTK](https://vtk.org/) files. First, we resample all the bundle files using `fibersampling.py` so all fibers have the same amount of points:
```bash
python fibersampling.py --num_points <new_number_of_points> [--csv | --path] <CSV-file-containing-tracts | Path to the folder containing vtk files> --output <output-path>
```
| <div style="width:150px">Argument</div> | Required | Description | Data Type |
|---|---|---|---|
| `--num_points` | Required | Number of points to resample the fiber with | Int |
| `--csv` | Required if `--path` not specified | Path to the CSV file containing the paths to tract files | Str |
| `--path` | Required if `--csv` not specified | Path to the folder containing the VTK files to resample | Str |
| `--output` | Required | Path of the output folder where resampled files will be saved | Str |

### Numpy conversion

Then we convert these files to [Numpy](https://numpy.org/) files using `vtk_to_numpy.py` because VTK is a slow computing library and it is faster to use `.npy` files:
```bash
python vtk_to_numpy.py --mode <single | multiple> --path <path-to-VTK-files> --num_points <num_points> --output <output-path>
```
| <div style="width:150px">Argument</div>| Required | Description | Data Type |
|---|---|---|---|
| `--mode` | Required | Mode to run the conversion `single` or `multiple`| Str |
| `--path` | Required | Path to the VTK files to convert (`mode` == `single`) or to the folders containing VTK files (`mode` == `multiple`) | Str |
| `--num_points` | Required | Number of points in a fiber (fibers must be resampled first) | Int |
| `--output` | Required | Path of the output folder where `.npy` files will be saved | Str |

Example of a `ls` command at given `--path` if `--mode` == `single`:
```bash
# Command
ls <path>

# Output
vtk_file1.vtk vtk_file2.vtk vtk_file3.vtk vtk_file4.vtk ... 
```
In the parent folder of `path` there has to be a `subject_brain.vtk` file.
Example of a `ls` command at given `--path` if `--mode` == `multiple`:
```bash
# Command
ls <path>

# Output
subject1 subject2 subject3 subject4 ... subject1_brain.vtk subject2_brain.vtk ...
```

### Compute CSV file

To generate the datasets, we use a Pytorch-Lightning **DataModule** that builds the **DataLoader** thanks to a `.csv` file generated with `computeCsv.py`.\
Here's how to compute the csv file:
```bash
python computeCsv.py--vtk_path <path-to-vtk-files> --npy_path <path-to-npy-files> --mode <tracts | brain> --num_subjects <single | multiple> --classes <path-to-classes-files> --name <output-file-name> --output <output-path>
```
| <div style="width:150px">Argument</div> | Required | Description | Data Type |
|---|---|---|---|
| `--vtk_path` | Required | Path to VTK files | Str |
| `--npy_path` | Required | Path to NPY files | Str |
| `--mode` | Required | Depending on the data you're computing the CSV file for, if the files are for labeled tracts use `tracts`, if the files are full brains use `brain` | Str|
| `--num_subjects` | Required | Number of subjects to process `single` or `multiple`. Default = `single` | Str |
| `--classes` | Required only if `mode` is `tracts` | Path to classes CSV file. This file has to contain the labels and the associated class | Str |
| `--name` | Optional | Name of the output CSV file. Default = `_output.csv` | Str |
| `--output` | Required | Path of the output folder where CSV file will be saved | Str |

Depending on the `mode`, the `vtk_path` and `npy_path` have to be the same structure as explained for Numpy conversion.\
Here's an example of a `classes` CSV file:
| label | class |
|:---:|:---:|
| 0 | "Class_0" |
| 1 | "Class_1" |
| 2 | "Class_2" |
| 3 | "Class_3" |
| 4 | "Class_4" |
|...|...|


For a better observation of the model's behaviour, we split the data using `splitData.py` to get **70% training**, **10% validation** and **20% testing** data:
```bash
python splitData.py --csv_file <path-to-csv-file> --test_size <test-size> --val_size <val-size> --random_state <random_state> --output <output-path>
```
| <div style="width:150px">Argument</div> | Required | Description | Data Type |
|---|---|---|---|
| `--csv_file` | Required | Path to the csv file | Str |
| `--test_size` | Required | Test size in percentage. Default = 0.2 | Float |
| `--val_size` | Required | Validation size in percentage. Default = 0.1 | Float |
| `--random_state` | optional | Random state. Default = 42 | Int |
| `--output` | Required | Path of the output folder where CSV files will be saved | Str |


## Getting Started : Used Models

We've tried multiple models :
* PointNet
* DEC
* SequentialDEC

To train one of them, you just have to run this command :
```bash
CUDA_VISIBLE_DEVICES=<GPU-you-want-to-train-on> python oneshot.py --config <config>
```
The config file can have 2 different versions, one for a single training and one for a knowledge transfer.\
Example of a single training:
```yaml
# General configuration
mode: half
pretrained_model: # If you want to transfer the knowledge from an other model, write the path of a checkpoint file
model: TractCurvNet
n_epochs: 500
num_workers: 7
k: 5
input_size: 4
embedding_size: 256
aggr: max
num_heads: 8

# Model & data configuration
train_path: YOUR_TRAINING_DATA.csv                # Path to the training data   
val_path: YOUR_VALIDATION_DATA.csv                # Path to the validation data
test_path: YOUR_TESTING_DATA.csv                  # Path to the test data
n_samples: 200                                    # Number of samples to be used
n_classes: 63                                     # Number of classes
batch_size: 4000                                  # Batch size
buffer_size: 500                                  # Buffer size
step_log: 10                                      # Step to log
checkpoint_path: YOUR_CHECKPOINT_PATH             # Path to save the model
dropout: True                                     # Use dropout
noise: True                                       # Use noise
noise_range: 0.1                                  # Noise range
rotation: True                                    # Use rotation
rotation_range: 0.1                               # Rotation range
shear: True                                       # Use shear
shear_range: 0.05                                 # Shear range
translation: False                                # Use translation
translation_range: 0.001                          # Translation range
augmentation_prob: 0.4                            # Augmentation probability

# Neptune configuration
api_key: YOUR_API_KEY
project: YOUR_PROJECT
monitor: val_loss
min_delta: 0.00
patience: 20
early_stopping_mode: min
```
If you want to transfer the knowledge from an other model, write the path to the trained model's checkpoint file in `pretrained_model`.

Example of a double traning (knowledge transfer):
```yaml
# General configuration
mode: full
model: TractCurvNet
n_epochs: 500
num_workers: 7
k: 5
input_size: 4
embedding_size: 256
aggr: max
num_heads: 8

# First model & data configuration
train_path_1: YOUR_TRAINING_DATA_1.csv            # Path to the training data
val_path_1: YOUR_VALIDATION_DATA_1.csv            # Path to the validation data
test_path_1: YOUR_TESTING_DATA_1.csv              # Path to the test data
n_samples_1: 1500                                 # Number of samples to be used
batch_size_1: 4000                                # Batch size
buffer_size_1: 500                                # Buffer size
n_classes_1: 58                                   # Number of classes
step_log_1: 100                                   # Step to log
checkpoint_path_1: YOUR_CHECKPOINT_PATH_1         # Path to save the model
dropout_1: True                                   # Use dropout
noise_1: True                                     # Use noise
noise_range_1: 0.01                               # Noise range
rotation_1: True                                  # Use rotation
rotation_range_1: 0.1                             # Rotation range
shear_1: True                                     # Use shear
shear_range_1: 0.01                               # Shear range
translation_1: True                               # Use translation
translation_range_1: 0.01                         # Translation range
augmentation_prob_1: 0.3                          # Augmentation probability

# Second model & data configuration
train_path_2: YOUR_TRAINING_DATA_2.csv                
val_path_2: YOUR_VALIDATION_DATA_2.csv                    
test_path_2: YOUR_TESTING_DATA_2.csv                  
n_samples_2: 200
batch_size_2: 4000
buffer_size_2: 500
n_classes_2: 63
step_log_2: 10
checkpoint_path_2: YOUR_CHECKPOINT_PATH_2
dropout_2: True
noise_2: True
noise_range_2: 0.01
rotation_2: True
rotation_range_2: 0.1
shear_2: True
shear_range_2: 0.01
translation_2: True
translation_range_2: 0.01
augmentation_prob_2: 0.5

# Neptune configuration
api_key: YOUR_API_KEY
project: YOUR_PROJECT
monitor: val_loss
min_delta: 0.00
patience: 10
early_stopping_mode: min
```

To visualize the training results, we use [neptune.ai](https://github.com/neptune-ai/neptune-client.git) which  is a lightweight experiment tracker for ML teams that struggle with debugging and reproducing experiments, sharing results, and messy model handover.\
Feel free to to set your own API_KEY and your own project file. You can also use an anonymous API

## Classification & Confidence level

To get a better classification, we've added a confidence branch to our model.\
Users can now use their model using a `.csv` file with all the **_classes_** and **_labels_** they want to classify and also adjust the confidence threshold they want to apply to each bundle in order to get a more or less selective classification.

The CSV file has to have this structure :

| label | class | confidence_thresholds |
|:---:|:---:|:---:|
| 0 | "Class_0" | 0.5 |
| 1 | "Class_1" | 0.2 |
| 2 | "Class_2" | 0.9 |
| 3 | "Class_3" | 0.0 |
| 4 | "Class_4" | 0.3 |
|...|...|...|

## Run a Classification

To run a classification, execute `run_classification.py` script with its options :

```bash
CUDA_VISIBLE_DEVICES=<GPU-you-want-to-run-on> python run_classification.py -h --model <PN, PNConf, BLSTM, DEC, DECConf, seqDEC, TractCurvNet> --checkpoint_path <PATH_TO_CHECKPOINT> --batch_size <BATCH_SIZE> --num_workers <NUMBER_OF_WORERS> --num_points <NUMBER_OF_SAMPLING_POINTS> --classes <CLASSES_JSON_FILE> --path <VTK_FILES_PATH> --output <OUTPUT_PATH>
```

Table explaining all arguments :

| <div style="width:150px">Argument</div> | Required | Description | Data Type |
|---|---|---|---|
| `--model` | Required | Model you want to run the classification {`PN`, `DEC`, `seqDEC`, `TractCurvNet`} | String |
| `--checkpoint_path` | Required | Path to the checkpoint file of a trained model | String |
| `--batch_size` | Required | Batch size for classification. Default = 500 | Int |
| `--num_workers` | Required | Number of workers for classification. Default = 7 | Int |
| `--num_points` | Required | Number of points to resample the data. Default = 128 | Int |
| `--classes` | Required | Path to CSV file containing the labels, classes and confidence thresholds | String |
| `--path` | Required | Path to the folder containing the brains' VTK files | String |
| `--output` | Required | Path where data will be saved | String









