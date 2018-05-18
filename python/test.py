from util import util
from random import shuffle
import os
import json

with open("data/training/experiment_3/train.json", "r") as train_json, open("data/training/experiment_3/val.json", "r") as val_json:
	list1 = json.load(train_json)
	list2 = json.load(val_json)
	print(set(["a", "b"]) & set(["a"]))
