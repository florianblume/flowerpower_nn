import numpy as np
import json

def reduce_sets(train, val, ratio):
    with open(train, "r") as train_file:
        train_data = json.load(train_file)
        train_data = np.array(train_data)
        indices = np.arange(0, len(train_data), ratio)
        indices = np.array([int(index) for index in indices])

    with open(train, "w") as train_file:
        new_train_data = train_data[indices]
        json.dump(list(new_train_data), train_file)

    with open(val, "r") as val_file:
        val_data = json.load(val_file)
        val_data = np.array(val_data)
        indices = np.arange(0, len(val_data), ratio)
        indices = np.array([int(index) for index in indices])
    
    with open(val, "w") as val_file:
        new_val_data = val_data[indices]
        json.dump(list(new_val_data), val_file)

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='This script provides functionality to visualize errors.')
    parser.add_argument("--train",
                        required=True,
                        help="The path to the training set json file.")
    parser.add_argument("--val",
                        required=True,
                        help="The path to the validation set json file.")
    parser.add_argument("--ratio",
                        required=False,
                        type=float,
                        help="The ration by which to reduce the sets.")
    args = parser.parse_args()
    reduce_sets(args.train, args.val, args.ratio)