import pandas as pd 
import os
import argparse
from sklearn.model_selection import train_test_split

def splitData(csvFile, output, test_size=0.2, val_size=0.1, random_state=42):
    """
    Split the data into train, validation and test sets based on subjects.
    """
    # Read the csv file
    data = pd.read_csv(csvFile)

    # Get unique subjects
    subjects = data['id'].unique()

    # Split the subjects into train and test sets
    train_subjects, test_subjects = train_test_split(subjects, test_size=test_size, random_state=random_state)

    # Split the train subjects into train and validation sets
    train_subjects, val_subjects = train_test_split(train_subjects, test_size=val_size, random_state=random_state)

    # Filter the data based on the selected subjects
    train_data = data[data['id'].isin(train_subjects)]
    val_data = data[data['id'].isin(val_subjects)]
    test_data = data[data['id'].isin(test_subjects)]

    # Save the data into csv files
    train_path = os.path.join(output, "_train.csv")
    val_path = os.path.join(output, "_val.csv")
    test_path = os.path.join(output, "_test.csv")

    train_data.to_csv(train_path, index=False)
    val_data.to_csv(val_path, index=False)
    test_data.to_csv(test_path, index=False)

    return train_path, val_path, test_path

def main(args):
    print("CSV file: ", args.csv_file)
    print("Test size: ", args.test_size)
    print("Validation size: ", args.val_size)
    print("Random state: ", args.random_state)

    train_path, val_path, test_path = splitData(args.csv_file, args.output, args.test_size, args.val_size, args.random_state)

    print("Train path: ", train_path)
    print("Val path: ", val_path)
    print("Test path: ", test_path)

def get_argparse():
    parser = argparse.ArgumentParser(description='Split data into train, validation and test sets', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--csv_file', type=str, help='Path to the csv file', default="/work/luvallad/_tracts.csv")
    parser.add_argument('--test_size', type=float, help='Test size', default=0.2)
    parser.add_argument('--val_size', type=float, help='Validation size', default=0.1)
    parser.add_argument('--random_state', type=int, help='Random state', default=42)
    parser.add_argument('--output', type=str, help='Output path', default=".")

    return parser

if __name__ == "__main__":
    parser = get_argparse()
    args = parser.parse_args()
    main(args)