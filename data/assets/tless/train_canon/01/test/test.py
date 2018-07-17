import json
from random import shuffle

for i in range(10):
    c = (i + 1) * 25
    with open("images_" + str(c) + ".json", "r") as images_file:
        images = json.load(images_file)
        shuffle(images)
        train_images = images[:int(len(images) * 0.7)]
        val_images = images[int(len(images) * 0.7)]
        with open("train_" + str(c) + ".json", "w") as train_file:
            json.dump(train_images, train_file)
        with open("val_" + str(c) + ".json", "w") as val_file:
            json.dump(val_images, val_file)