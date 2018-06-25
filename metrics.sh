#!/bin/bash
model=$1
experiment=$2

eval $"python python/metrics.py --gt_images data/assets/tless/train_canon/01/generated/obj_coords/ --pred_images data/experiments/${model}/${experiment}/inference_val/predictions/ --gt data/assets/tless/train_canon/01/gt.json --pred data/experiments/${model}/${experiment}/inference_val/inferred_poses.json --obj obj_01 --image_extension jpg --output_file data/experiments/${model}/${experiment}/inference_val/metrics.json"
eval $"python python/metrics.py --gt_images data/assets/tless/train_canon/01/generated/obj_coords/ --pred_images data/experiments/${model}/${experiment}/inference_test/predictions/ --gt data/assets/tless/train_canon/01/gt.json --pred data/experiments/${model}/${experiment}/inference_test/inferred_poses.json --obj obj_01 --image_extension jpg --output_file data/experiments/${model}/${experiment}/inference_test/metrics.json"