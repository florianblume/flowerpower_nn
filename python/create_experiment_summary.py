import os
import json
from util import util
from collections import OrderedDict

def gather_summary(exp_path, output_path):
    with open(output_path, "w") as output_file:
        output_data = OrderedDict()
        models = [name for name in os.listdir(exp_path) if os.path.isdir(os.path.join(exp_path, name))]
        util.sort_list_by_num_in_string_entries(models)
        for model in models:
            output_data[model] = OrderedDict()
            model_path = os.path.join(exp_path, model)
            object_models = [name for name in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, name))]
            util.sort_list_by_num_in_string_entries(object_models)
            for object_model in object_models:
                output_data[model][object_model] = OrderedDict()
                object_model_path = os.path.join(model_path, object_model)
                experiments = [name for name in os.listdir(object_model_path) if os.path.isdir(os.path.join(object_model_path, name))]
                util.sort_list_by_num_in_string_entries(experiments)
                for experiment in experiments:
                    output_data[model][object_model][experiment] = OrderedDict()
                    metrics_path = os.path.join(object_model_path, experiment, "inference_val", "metrics.json")
                    if os.path.exists(metrics_path):
                        with open(metrics_path, "r") as metrics_file:
                            metrics_data = json.load(metrics_file)
                            output_data[model][object_model][experiment]["mean"] = metrics_data["mean"]
                            output_data[model][object_model][experiment]["median"] = metrics_data["median"]
                    else:
                        print("No metrics.json found for {}.".format(model + " - " + object_model + " - " + experiment))
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