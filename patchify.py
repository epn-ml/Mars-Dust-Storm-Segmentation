# -*- coding: utf-8 -*-

from keras.utils import normalize
import os
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import random
import math
import shutil
import pandas as pd

############################################################################
#ALL DUST IMAGES - patching 1024x1024
##############################################################################

TRAIN_PATH = "./images/"
PATCH_PATH = './all_1024_img_patches/'

patch_size = 1024

for folder in next(os.walk(TRAIN_PATH))[1]:
    for sub_folder in next(os.walk(TRAIN_PATH + folder))[1]:
        for img_file in next(os.walk(TRAIN_PATH + folder + "/" + sub_folder))[2]:
            path = TRAIN_PATH + folder + "/" + sub_folder + "/" + img_file
            img = cv2.imread(path)
            img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
            for i in range(2):   #Steps of 256
                for j in range(4):  #Steps of 256
                    if j != 3:
                        if i == 0:
                            single_patch = img[0:patch_size, j*patch_size:(j+1)*patch_size:]
                            cv2.imwrite(PATCH_PATH + 'image_' + img_file[0:9] + '_' + str(i) + str(j) + '.png', cv2.cvtColor(single_patch, cv2.COLOR_RGB2BGR))
                        if i == 1:
                            single_patch = img[777:, j*patch_size:(j+1)*patch_size:]
                            cv2.imwrite(PATCH_PATH + 'image_' + img_file[0:9] + '_' + str(i) + str(j) + '.png', cv2.cvtColor(single_patch, cv2.COLOR_RGB2BGR))
    
                    if j == 3:
                        if i == 0:
                            single_patch = img[0:patch_size, 2576::]
                            cv2.imwrite(PATCH_PATH + 'image_' + img_file[0:9] + '_' + str(i) + str(j) + '.png', cv2.cvtColor(single_patch, cv2.COLOR_RGB2BGR))
                        if i == 1:
                            single_patch = img[777:, 2576::]
                            cv2.imwrite(PATCH_PATH + 'image_' + img_file[0:9] + '_' + str(i) + str(j) + '.png', cv2.cvtColor(single_patch, cv2.COLOR_RGB2BGR))


TRAIN_PATH_MASK = "./masks/"
MASK_PATH = './all_1024_mask_patches/'

patch_size = 1024

for folder in next(os.walk(TRAIN_PATH_MASK))[1]:
    for sub_folder in next(os.walk(TRAIN_PATH_MASK + folder))[1]:
        for img_file in next(os.walk(TRAIN_PATH_MASK + folder + "/" + sub_folder))[2]:
            path = TRAIN_PATH_MASK + folder + "/" + sub_folder + "/" + img_file
            img = cv2.imread(path, 0)
            #img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
            for i in range(2):   #Steps of 256
                for j in range(4):  #Steps of 256
                    if j != 3:
                        if i == 0:
                            single_patch = img[0:patch_size, j*patch_size:(j+1)*patch_size]
                            cv2.imwrite(MASK_PATH + 'image_' + img_file[0:9] + '_' + str(i) + str(j) + '.png', single_patch)
                        if i == 1:
                            single_patch = img[777:, j*patch_size:(j+1)*patch_size]
                            cv2.imwrite(MASK_PATH + 'image_' + img_file[0:9] + '_' + str(i) + str(j) + '.png', single_patch)
    
                    if j == 3:
                        if i == 0:
                            single_patch = img[0:patch_size, 2576:]
                            cv2.imwrite(MASK_PATH + 'image_' + img_file[0:9] + '_' + str(i) + str(j) + '.png', single_patch)
                        if i == 1:
                            single_patch = img[777:, 2576:]
                            cv2.imwrite(MASK_PATH + 'image_' + img_file[0:9] + '_' + str(i) + str(j) + '.png', single_patch)

