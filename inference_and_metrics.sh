#!/bin/bash
model=$1
experiment=$2

eval $"python python/inference_pos.py --config data/experiments/${model}/${experiment}/inference_val/config.json"
eval $"python python/inference_raw.py --config data/experiments/${model}/${experiment}/inference_val/config.json"
eval $"python python/inference_pos.py --config data/experiments/${model}/${experiment}/inference_test/config.json"
eval $"python python/inference_raw.py --config data/experiments/${model}/${experiment}/inference_test/config.json"
eval $"python python/util/visualize_errors.py --gt data/assets/tless/train_canon/01/generated/obj_coords/ --pred data/experiments/${model}/${experiment}/inference_val/predictions/ --output_path data/experiments/${model}/${experiment}/inference_val/errors/"
eval $"python python/metrics.py --gt_images data/assets/tless/train_canon/01/generated/obj_coords/ --pred_images data/experiments/${model}/${experiment}/inference_val/predictions/ --gt data/assets/tless/train_canon/01/gt.json --pred data/experiments/${model}/${experiment}/inference_val/inferred_poses.json --obj obj_01.ply --image_extension jpg --output_file data/experiments/${model}/${experiment}/inference_val/metrics.json"