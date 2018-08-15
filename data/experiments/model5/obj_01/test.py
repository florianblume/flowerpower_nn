import json
import numpy as np
from random import randint

indices = np.arange(0, 1296, 5.184)
indices = indices.astype(np.int32)

images = [str(i).zfill(4) + '.jpg' for i in np.arange(1296)]

d = [[],[],[],[],[],[],[],[],[],[]]

for i in range(indices.shape[0]):
    d[i%10].append(images[indices[i]])

val_images = []
train_total = []

for i, images in enumerate(d):
    index = i + 1
    i_0 = randint(0, 2)
    i_1 = randint(3, 5)
    i_2 = randint(6, 8)
    i_3 = randint(9, 11)
    i_4 = randint(12, 14)
    i_5 = randint(15, 17)
    i_6 = randint(19, 21)
    i_7 = randint(22, 24)
    i_s = [i_0, i_1, i_2, i_3, i_4, i_5, i_6, i_7]
    train_images = images[0:i_0]
    for i_x in range(len(i_s)):
        val_images.append(images[i_s[i_x]])
        if i_x + 1 < len(i_s):
            train_images = train_images + images[i_s[i_x] + 1: i_s[i_x + 1]]
    train_images = train_images + images[i_7 + 1:25]
    train_total = train_total + train_images
    print(train_images)
    with open('experiment_7_' + str(index) + '/training/train.json', 'w') as train_file:
        json.dump(train_images, train_file)
    with open('experiment_7_' + str(index) + '/training/val.json', 'w') as val_file:
        json.dump(val_images, val_file)

total = val_images + train_total
d = np.array(d)
d = d.squeeze()
print([image for image in total if image not in d])
print([image for image in val_images if image in train_total])