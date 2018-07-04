import os
import json
from collections import OrderedDict

def gather_summary(exp_path, output_path):
    with open(output_path, "w") as output_file:
        output_data = OrderedDict()
        models = [name for name in os.listdir(exp_path) if os.path.isdir(os.path.join(exp_path, name))]
        for model in models:
            model_path = os.path.join(exp_path, model)
            experiments = [name for name in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, name))]
            for experiment in experiments:
                key = model + "/" + experiment
                output_data[key] = {}
                metrics_path = os.path.join(model_path, experiment, "inference_val", "metrics.json")
                with open(metrics_path, "r") as metrics_file:
                    metrics_data = json.load(metrics_file)
                    output_data[key]["mean"] = metrics_data["mean"]
                    output_data[key]["median"] = metrics_data["median"]
        json.dump(output_data, output_file)


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='This script provides functionality to calculate metrics for a set of '
                    'object coordinate predictions and ground truth object coordinates.')
    parser.add_argument("--exp_path",
                        required=True,
                        help="The path to the experiment folders.")
    parser.add_argument("--output_path",
                        required=True,
                        help="The file where to store the results.")
    args = parser.parse_args()
    gather_summary(args.exp_path, args.output_path)