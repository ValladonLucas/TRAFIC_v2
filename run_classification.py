import argparse
from Classification import classification
from Data_processing import fibersampling, computeCsv
import os
import shutil
import pandas as pd

def get_argparse_dict(parser):
    # Get the default arguments from the parser
    default = {}
    for action in parser._actions:
        if action.dest != "help":
            default[action.dest] = action.default
    return default

def main(args):
    print("""\n################################ Starting the classification process ################################\n""")

    df = pd.DataFrame(columns=["id", "original_path", "sampled_path"])
    for file in os.listdir(args.path):
        if file.endswith(".vtk"):
            df.loc[len(df)] = [file.split('.')[0], os.path.join(args.path, file), None]
            # df = df._append({"original_path": os.path.join(args.path, file), "id": file.split('.')[0]}, ignore_index=True)

    if not os.path.exists(args.output):
        os.mkdir(args.output)

    ############################ 1. Resample the data ############################
    print("""\n################################ Resampling the data ################################\n""")
    fibersampling_args = get_argparse_dict(fibersampling.get_argparse())
    fibersampling_args["num_points"] = args.num_points
    fibersampling_args["path"] = args.path
    fibersampling_args["output"] = os.path.join(args.output, f"sampled_{fibersampling_args['num_points']}")

    if not os.path.exists(fibersampling_args["output"]):
        os.mkdir(fibersampling_args["output"])

    fibersampling.main(argparse.Namespace(**fibersampling_args))

    for file in os.listdir(fibersampling_args["output"]):
        if file.endswith(".vtk"):
            df.loc[df["id"] == file.split('.')[0].replace("_sampled",""), "sampled_path"] = os.path.join(fibersampling_args["output"], file)
    df.to_csv(os.path.join(args.output, "brains.csv"), index=False)

    ############################ 2. Make the csv file ############################
    print("""\n################################ Making the csv file ################################\n""")
    computeCsv_args = get_argparse_dict(computeCsv.get_argparse())
    computeCsv_args["vtk_path"] = f"{fibersampling_args['output']}"
    computeCsv_args["output"] = f"{fibersampling_args['path']}"
    computeCsv_args["name"] = f"sampled_{fibersampling_args['num_points']}_output.csv"
    computeCsv_args["num_subjects"] = "single"
    computeCsv_args["mode"] = "brain"

    computeCsv.main(argparse.Namespace(**computeCsv_args))

    ############################ 3. Classify the data ############################
    print("""\n################################ Classifying the data ################################\n""")
    print(f"\n Used model : {args.checkpoint_path.split('/')[-1]}\n")
    classification_args = get_argparse_dict(classification.get_argparse())
    classification_args["csv_path"] = f"{computeCsv_args['output']}/{computeCsv_args['name']}"
    classification_args["original_brain"] = os.path.join(args.output, "brains.csv")
    classification_args["batch_size"] = args.batch_size
    classification_args["num_workers"] = args.num_workers
    classification_args["num_points"] = args.num_points
    classification_args["num_classes"] = len(pd.read_csv(args.classes)['label'])
    classification_args["model"] = args.model
    classification_args["checkpoint_path"] = args.checkpoint_path
    classification_args["save_path"] = os.path.join(args.output, "predicted_tracts")
    classification_args["classes"] = args.classes
    classification_args["k"] = 5

    if not os.path.exists(classification_args["save_path"]):
        os.mkdir(classification_args["save_path"])
    else:
        # Clear the save path folder
        shutil.rmtree(classification_args["save_path"])
        os.mkdir(classification_args["save_path"])

    classification.main(argparse.Namespace(**classification_args))


def get_argparse():
    parser = argparse.ArgumentParser(description='Whole process to classification', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', type=int, help='Batch size', default=500)
    parser.add_argument('--num_workers', type=int, help='Number of workers', default=7)
    parser.add_argument('--num_points', type=int, help='Number of points', default=128)

    parser.add_argument('--model', type=str, help='Model to train', default="TractCurvNet")
    parser.add_argument('--checkpoint_path', type=str, help='Path to the checkpoint', required=True)

    parser.add_argument('--classes', type=str, help='Path to the json file with the classes', required=True)

    parser.add_argument('--path', type=str, help='Path to vtk file', required=True)
    parser.add_argument('--output', type=str, help='Path to save the output', required=True)

    return parser

if __name__ == "__main__":
    args = get_argparse().parse_args()
    main(args)