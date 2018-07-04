import os
import json

def get_best_experiment(summary):
    best = {"mean" : {"inliers" : {"value" : -1, "experiment" : ""}, 
                      "pixel_error" : {"value" : -1, "experiment" : ""}, 
                      "angle_error" : {"value" : -1, "experiment" : ""}, 
                      "distance_error" : {"value" : -1, "experiment" : ""}}, 
            "median" : {"25" : {}, "50": {}, "75": {}}}
    with open(summary, "r") as summary_file:
        summary_data = json.load(summary_file)
        for experiment in summary_data:
            experiment_data = summary_data[experiment]
            print(experiment_data)
            experiment_mean  = experiment_data["mean"]
            if best["mean"]["pixel_error"]["value"] == -1 or best["mean"]["pixel_error"]["value"] > experiment_mean["pixel_error"]:
                best["mean"]["pixel_error"]["value"] = experiment_mean["pixel_error"]
                best["mean"]["pixel_error"]["experiment"] = experiment
            if best["mean"]["inliers"]["value"] == -1 or best["mean"]["inliers"]["value"] > experiment_mean["inliers"]:
                best["mean"]["inliers"]["value"] = experiment_mean["inliers"]
                best["mean"]["inliers"]["experiment"] = experiment
            if best["mean"]["angle_error"]["value"] == -1 or best["mean"]["angle_error"]["value"] > experiment_mean["angle_error"]:
                best["mean"]["angle_error"]["value"] = experiment_mean["angle_error"]
                best["mean"]["angle_error"]["experiment"] = experiment
            if best["mean"]["distance_error"]["value"] == -1 or best["mean"]["distance_error"]["value"] > experiment_mean["distance_error"]:
                best["mean"]["distance_error"]["value"] = experiment_mean["distance_error"]
                best["mean"]["distance_error"]["experiment"] = experiment
            experiment_median  = experiment_data["median"]
        print(best)


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='This script provides functionality to get the best experiment from'
                    ' a experiments summary file.')
    parser.add_argument("--summary",
                        required=True,
                        help="The summary file to compute the best experiment from.")
    args = parser.parse_args()
    get_best_experiment(args.summary)