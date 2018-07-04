#!/bin/bash
model=$1
experiment=$2

eval $"python python/inference_raw.py --config ./data/experiments/${model}/${experiment}/inference_val/config.json"
eval $"python python/inference_pos.py --config ./data/experiments/${model}/${experiment}/inference_val/config.json"
eval $"python python/inference_raw.py --config ./data/experiments/${model}/${experiment}/inference_test/config.json"
eval $"python python/inference_pos.py --config ./data/experiments/${model}/${experiment}/inference_test/config.json"
