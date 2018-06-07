from util import util
from random import shuffle
import os
import json
import cv2

import numpy as np
import tifffile as tiff

def compute_reprojection(prediction, rvec, tvec, cam_info):
    reprojection, _ = cv2.projectPoints(prediction,
                                        rvec,
                                        tvec,
                                        np.array(cam_info['K']).reshape(3, 3),
                                        None)
    return reprojection

predicted = tiff.imread("/home/florian/git/flowerpower_nn/data/experiments/experiment_4/inference/predictions/obj_coords_0500.tiff")
predicted = predicted.astype(np.float32)
gt = tiff.imread("/home/florian/git/flowerpower_nn/data/assets/tless/train_canon/01/generated/obj_coords/obj_coords_0500.tiff")
gt = gt.astype(np.float32)
gt = util.shrink_image_with_step_size(gt, predicted.shape)

diff = np.absolute(predicted - gt)
print(diff)

indices = np.where((diff < 0.1) & (diff > 0))

cam_info_file = open("/home/florian/git/flowerpower_nn/data/assets/tless/train_canon/01/generated/info.json", "r")
cam_info = json.load(cam_info_file)["0500.jpg"]

objs = gt[indices[0], indices[1]]
gts = gt[indices[0], indices[1]]
imgs = np.stack([indices[1], indices[0]], axis=1)
for index in range(min(objs.shape[0], 100)):
    print("{} {} {}".format(imgs[index], objs[index], gts[index]))

imgs = imgs.astype(np.float32)
retval, rvec, tvec, inliers  = cv2.solvePnPRansac(objs, 
                              imgs, 
                              np.array(cam_info['K']).reshape(3, 3), 
                              None,
                              iterationsCount=1000
    )

reprojection = np.squeeze(compute_reprojection(objs, rvec, tvec, cam_info))
print("r: {}".format(rvec))
print("t: {}".format(tvec))
print(imgs.shape)
print(reprojection.shape)
print("Pose error {}".format(np.mean(np.absolute(imgs - reprojection))))