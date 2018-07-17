import json
from random import shuffle
import numpy as np
import re

with open("experiment_1/training/val.json") as val_file:
    val_data = json.load(val_file)

with open("experiment_1/training/train.json") as train_file:
    train_data = json.load(train_file)

images = val_data + train_data
images.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
i = np.arange(0, len(images), 5.184)
i = i.astype(np.int32)

partitions = [[],[],[],[],[],[],[],[],[],[]]

for j in range(len(i)):
    index = i[j]
    partitions[j % 10].append(images[index])

with open("experiment_7_25/training/train_25.json", "w") as train, \
     open("experiment_7_25/training/val_25.json", "w") as val:
     data = partitions[0]
     shuffle(data)
     train_data = data[:int(len(data) * 0.7)]
     val_data = data[int(len(data) * 0.7):]
     json.dump(train_data, train)
     json.dump(val_data, val)

for i in range(1, 10):
    before = i - 1
    data = partitions[i]
    shuffle(data)
    train_data = data[:int(len(data) * 0.7)]
    val_data = data[int(len(data) * 0.7):]
    with open("experiment_7_" + str(i * 25) + "/training/val_" + str(i * 25) + ".json", "r") as val_before:
        val_before_data = json.load(val_before)
    with open("experiment_7_" + str((i + 1) * 25) + "/training/val_" + str((i + 1) * 25) + ".json", "w") as val_next:
        val_next_data = val_before_data + val_data
        json.dump(val_next_data, val_next)
    with open("experiment_7_" + str((i + 1) * 25) + "/training/train_" + str((i + 1) * 25) + ".json", "w") as train:
        json.dump(train_data, train)