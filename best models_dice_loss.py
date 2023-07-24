# -*- coding: utf-8 -*-

import segmentation_models as sm
import keras 
from keras.metrics import MeanIoU
from io import BytesIO

import tensorflow as tf
import os
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.optimizers import Adam
from datetime import datetime 
import cv2
from PIL import Image
from numpy import asarray
from numpy import save
from numpy import load
import pandas as pd


np.random.seed(0)
tf.random.set_seed(0)


RESIZED_SIZE = 512

from keras.models import load_model

#Set compile=False as we are not loading it for training, only for prediction.
model1 = load_model('./iou loss/models/unet_2d_ResNet50_100epochs_G05_1024patchsize_IOULoss.h5', compile=False)
model2 = load_model('./iou loss/models/unet_2d_DenseNet121_100epochs_G05_1024patchsize_IOULoss.h5', compile=False)
model3 = load_model('./iou loss/models/att_unet_2d_DenseNet201_100epochs_G05_1024patchsize_IOULoss.h5', compile=False)
model4 = load_model('./iou loss/models/unet_plus_2d_ResNet152_100epochs_G05_1024patchsize_IOULoss.h5', compile=False)
model5 = load_model('./iou loss/models/unet_inceptionv3_100epochs_G05_1024patchsize_IOULoss.h5', compile=False)
model6 = load_model('./iou loss/models/unet_inceptionv3_100epochs_G05G04_512patchsize_IOULoss.h5', compile=False)
model7 = load_model('./iou loss/models/unet_vgg16_100epochs_G05G04_512patchsize_IOULoss.h5', compile=False)


##############################################################

BACKBONE5 = 'inceptionv3'
preprocess_input5 = sm.get_preprocessing(BACKBONE5)

BACKBONE6 = 'inceptionv3'
preprocess_input6 = sm.get_preprocessing(BACKBONE6)

BACKBONE7 = 'vgg16'
preprocess_input7 = sm.get_preprocessing(BACKBONE7)


test_image_path = './dust_1024_patch_size/test_img/'
test_mask_path = './dust_1024_patch_size/test_mask/'
test_color_path = "./dust_1024_patch_size/test_mask_color_coded/"

test_image_dataset = []  
test_mask_dataset = [] 
test_color_dataset = []

images = os.listdir(test_image_path)
for i, image_name in enumerate(images):   
    image = cv2.imread(test_image_path+image_name)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    test_image_dataset.append(np.array(image))


masks = os.listdir(test_mask_path)
for i, image_name in enumerate(masks):
    image = cv2.imread(test_mask_path+image_name, 0)
    image = Image.fromarray(image)
    test_mask_dataset.append(np.array(image))
    
    
images = os.listdir(test_color_path)
for i, image_name in enumerate(images):   
    image = cv2.imread(test_color_path+image_name)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    test_color_dataset.append(np.array(image))
    
test_image_dataset = np.array(test_image_dataset)
test_mask_dataset = np.array(test_mask_dataset)
test_color_dataset = np.array(test_color_dataset)

def calculate_iou(pred_mask, gt_mask):
    if gt_mask is None:
        return 0.0

    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    if (union != 0):
        iou = intersection / union
    else:
        iou = 1

    return iou


# create a list of models and preprocessing functions
models = [model1, model2, model3, model4]
#preprocess_functions = [preprocess_input1, preprocess_input2, preprocess_input3]

# create an empty dataframe to store the IOU values
iou_df = pd.DataFrame()

# loop through each model and test image, and add a new column to the dataframe for each model's IOU values
for model_idx, model in enumerate(models):
    iou_values = []
    for i in range(len(test_image_dataset)):
        # Use the model to generate a segmentation mask for the test image
        test_img = test_image_dataset[i]
        #ground_truth=y_test[test_img_number]
        test_img_input = np.expand_dims(test_img, 0)
        
        #preprocess_input = preprocess_functions[model_idx]
        #test_img_input = preprocess_input(test_img_input)
        
        test_pred = model.predict(test_img_input)
        test_prediction = np.argmax(test_pred, axis=3)[0,:,:]
        
        
        gt_mask = test_mask_dataset[i]
        gt_mask = gt_mask/255.
        
        # Calculate the IOU for this test image
        iou = calculate_iou(test_prediction, gt_mask)
        iou_values.append(iou)
    
    # add a new column to the dataframe for this model's IOU values
    iou_df[f'model{model_idx+1}'] = iou_values


test_image_dataset_5 = []  
test_mask_dataset_5 = []

images_5 = os.listdir(test_image_path)
for i, image_name in enumerate(images_5):   
    image = cv2.imread(test_image_path+image_name)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    test_image_dataset_5.append(np.array(image))


masks_5 = os.listdir(test_mask_path)
for i, image_name in enumerate(masks_5):
    image = cv2.imread(test_mask_path+image_name, 0)
    #image = cv2.resize(image, (RESIZED_SIZE, RESIZED_SIZE), interpolation=cv2.INTER_NEAREST)
    image = Image.fromarray(image)
    #image = image.resize((SIZE, SIZE))
    test_mask_dataset_5.append(np.array(image))
    

test_image_dataset_5 = np.array(test_image_dataset_5)
test_mask_dataset_5 = np.array(test_mask_dataset_5)

def calculate_iou(pred_mask, gt_mask):
    if gt_mask is None:
        return 0.0

    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    if (union != 0):
        iou = intersection / union
    else:
        iou = 1

    return iou


# create a list of models and preprocessing functions
models = [model5]
preprocess_functions = [preprocess_input5]

# create an empty dataframe to store the IOU values
iou_df_5 = pd.DataFrame()

# loop through each model and test image, and add a new column to the dataframe for each model's IOU values
for model_idx, model in enumerate(models):
    iou_values = []
    for i in range(len(test_image_dataset_5)):
        # Use the model to generate a segmentation mask for the test image
        test_img = test_image_dataset_5[i]
        #ground_truth=y_test[test_img_number]
        test_img_input = np.expand_dims(test_img, 0)
        
        preprocess_input = preprocess_functions[model_idx]
        test_img_input = preprocess_input(test_img_input)
        
        test_pred = model.predict(test_img_input)
        test_prediction = np.argmax(test_pred, axis=3)[0,:,:]
        
        
        gt_mask = test_mask_dataset_5[i]
        gt_mask = gt_mask/255.
        
        # Calculate the IOU for this test image
        iou = calculate_iou(test_prediction, gt_mask)
        iou_values.append(iou)
    
    # add a new column to the dataframe for this model's IOU values
    iou_df_5[f'model{model_idx+1}'] = iou_values


test_image_dataset_67 = []  
test_mask_dataset_67 = [] 

images_67 = os.listdir(test_image_path)
for i, image_name in enumerate(images_67):   
    image = cv2.imread(test_image_path+image_name)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (RESIZED_SIZE, RESIZED_SIZE))
    image = Image.fromarray(image)
    test_image_dataset_67.append(np.array(image))

masks_67 = os.listdir(test_mask_path)
for i, image_name in enumerate(masks_67):
    image = cv2.imread(test_mask_path+image_name, 0)
    image = cv2.resize(image, (RESIZED_SIZE, RESIZED_SIZE), interpolation=cv2.INTER_NEAREST)
    image = Image.fromarray(image)
    #image = image.resize((SIZE, SIZE))
    test_mask_dataset_67.append(np.array(image))
    
test_image_dataset_67 = np.array(test_image_dataset_67)
test_mask_dataset_67 = np.array(test_mask_dataset_67)

def calculate_iou(pred_mask, gt_mask):
    if gt_mask is None:
        return 0.0

    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    if (union != 0):
        iou = intersection / union
    else:
        iou = 1

    return iou


# create a list of models and preprocessing functions
models = [model6, model7]
preprocess_functions = [preprocess_input6, preprocess_input7]

# create an empty dataframe to store the IOU values
iou_df_67 = pd.DataFrame()

# loop through each model and test image, and add a new column to the dataframe for each model's IOU values
for model_idx, model in enumerate(models):
    iou_values = []
    for i in range(len(test_image_dataset_67)):
        # Use the model to generate a segmentation mask for the test image
        test_img = test_image_dataset_67[i]
        #ground_truth=y_test[test_img_number]
        test_img_input = np.expand_dims(test_img, 0)
        
        preprocess_input = preprocess_functions[model_idx]
        test_img_input = preprocess_input(test_img_input)
        
        test_pred = model.predict(test_img_input)
        test_prediction = np.argmax(test_pred, axis=3)[0,:,:]
        
        
        gt_mask = test_mask_dataset_67[i]
        gt_mask = gt_mask/255.
        
        # Calculate the IOU for this test image
        iou = calculate_iou(test_prediction, gt_mask)
        iou_values.append(iou)
    
    # add a new column to the dataframe for this model's IOU values
    iou_df_67[f'model{model_idx+1}'] = iou_values


iou_df = iou_df.rename(columns={'model1': 'unet_2d_ResNet50_1024ps', 'model2': 'unet_2d_DenseNet121_1024ps', 
                                'model3': 'att_unet_2d_DenseNet201_1024ps', 'model4': 'unet_plus_2d_ResNet152_1024ps'})


iou_df_5 = iou_df_5.rename(columns={'model1': 'unet_inceptionv3_1024ps'})
iou_df_67 = iou_df_67.rename(columns={'model1': 'unet_inceptionv3_512ps', 'model2': 'unet_vgg16_512ps'})
iou_df_total = pd.concat([iou_df, iou_df_5, iou_df_67], axis=1)
iou_df_total.to_csv('./iou loss/iou_of_test_images.csv', index=False)


cnt = 0

test_labels = ["25_03", "25_10", "25_11", "25_12", "25_13", "26_00", "26_01", 
               "26_02", "26_03", "26_11", "26_12", "27_00", "27_01", "27_02", 
               "27_03", "27_10", "27_11", "28_00", "28_01", "28_02", "28_03", 
               "28_10", "28_11", "29_00", "29_01", "29_02", "29_03", "29_10", 
               "30_00", "30_01"]

for test_lbl in test_labels:
    
    color_mask = cv2.imread("./dust_1024_patch_size/test_mask_color_coded/image_G05_day" 
                            + test_lbl + ".png.png")
    color_mask = cv2.cvtColor(color_mask
                              ,cv2.COLOR_BGR2RGB)
    test_img = cv2.imread("./dust_1024_patch_size/test_img/image_G05_day" 
                          + test_lbl + ".png")
    test_img = cv2.cvtColor(test_img,cv2.COLOR_BGR2RGB)
    test_img_resized = cv2.resize(test_img, (RESIZED_SIZE, RESIZED_SIZE))

    ground_truth=cv2.imread("./dust_1024_patch_size/test_mask/image_G05_day" 
                          + test_lbl + ".png", 0)
    
    ground_truth_resized = cv2.resize(ground_truth, (RESIZED_SIZE, RESIZED_SIZE), interpolation=cv2.INTER_NEAREST)

    test_img=np.expand_dims(test_img, 0)
    
    test_img_resized=np.expand_dims(test_img_resized, 0)


    test_pred1 = model1.predict(test_img)
    test_prediction1 = np.argmax(test_pred1, axis=3)[0,:,:]

    test_pred2 = model2.predict(test_img)
    test_prediction2 = np.argmax(test_pred2, axis=3)[0,:,:]

    test_pred3 = model3.predict(test_img)
    test_prediction3 = np.argmax(test_pred3, axis=3)[0,:,:]

    test_pred4 = model4.predict(test_img)
    test_prediction4 = np.argmax(test_pred4, axis=3)[0,:,:]

    test_img_input5 = preprocess_input5(test_img)
    test_pred5 = model5.predict(test_img_input5)
    test_prediction5 = np.argmax(test_pred5, axis=3)[0,:,:]

    test_img_input6 = preprocess_input6(test_img_resized)
    test_pred6 = model6.predict(test_img_input6)
    test_prediction6 = np.argmax(test_pred6, axis=3)[0,:,:]
    
    test_img_input7 = preprocess_input7(test_img_resized)
    test_pred7 = model7.predict(test_img_input7)
    test_prediction7 = np.argmax(test_pred7, axis=3)[0,:,:]


    plt.figure(figsize=(40, 5))
    plt.subplot(1, 10, 1)
    #plt.title('Testing Image')
    plt.imshow(test_img.squeeze())
    plt.subplot(1, 10, 2)
    #plt.title('Testing Label')
    plt.imshow(ground_truth, cmap='gray')
    plt.xticks([])  # remove x-axis values
    #plt.yticks([])  # remove y-axis values
    plt.subplot(1, 10, 3)
    #plt.title('Testing Label color coded')
    plt.imshow(color_mask.squeeze())
    plt.xticks([])  # remove x-axis values
    #plt.yticks([])  # remove y-axis values
    plt.subplot(1, 10, 4)
    plt.title('unet_2d_ResNet50_1024ps')
    plt.imshow(test_prediction1)
    plt.text(0.5, -0.1, str(round(iou_df["unet_2d_ResNet50_1024ps"][cnt], 3)), ha='center', va='center', transform=plt.gca().transAxes, fontsize=35)
    plt.xticks([])  # remove x-axis values
    #plt.yticks([])  # remove y-axis values
    plt.subplot(1, 10, 5)
    plt.title('unet_2d_DenseNet121_1024ps')
    plt.imshow(test_prediction2)
    plt.text(0.5, -0.1, str(round(iou_df["unet_2d_DenseNet121_1024ps"][cnt], 3)), ha='center', va='center', transform=plt.gca().transAxes, fontsize=35)
    plt.xticks([])  # remove x-axis values
    #plt.yticks([])  # remove y-axis values
    plt.subplot(1, 10, 6)
    plt.title('att_unet_2d_DenseNet201_1024ps')
    plt.imshow(test_prediction3)
    plt.text(0.5, -0.1, str(round(iou_df["att_unet_2d_DenseNet201_1024ps"][cnt], 3)), ha='center', va='center', transform=plt.gca().transAxes, fontsize=35)
    plt.xticks([])  # remove x-axis values
    #plt.yticks([])  # remove y-axis values
    plt.subplot(1, 10, 7)
    plt.title('unet_plus_2d_ResNet152_1024ps')
    plt.imshow(test_prediction4)
    plt.text(0.5, -0.1, str(round(iou_df["unet_plus_2d_ResNet152_1024ps"][cnt], 3)), ha='center', va='center', transform=plt.gca().transAxes, fontsize=35)
    plt.xticks([])  # remove x-axis values
    #plt.yticks([])  # remove y-axis values
    plt.subplot(1, 10, 8)
    plt.title('unet_inceptionv3_1024ps')
    plt.imshow(test_prediction5)
    plt.text(0.5, -0.1, str(round(iou_df_5["unet_inceptionv3_1024ps"][cnt], 3)), ha='center', va='center', transform=plt.gca().transAxes, fontsize=35)
    plt.xticks([])  # remove x-axis values
    #plt.yticks([])  # remove y-axis values
    plt.subplot(1, 10, 9)
    plt.title('unet_inceptionv3_512ps')
    plt.imshow(test_prediction6)
    plt.text(0.5, -0.1, str(round(iou_df_67["unet_inceptionv3_512ps"][cnt], 3)), ha='center', va='center', transform=plt.gca().transAxes, fontsize=35)
    plt.xticks([])  # remove x-axis values
    #plt.yticks([])  # remove y-axis values
    plt.subplot(1, 10, 10)
    plt.title('unet_vgg16_512ps')
    plt.imshow(test_prediction7)
    plt.text(0.5, -0.1, str(round(iou_df_67["unet_vgg16_512ps"][cnt], 3)), ha='center', va='center', transform=plt.gca().transAxes, fontsize=35)
    plt.xticks([])  # remove x-axis values
    #plt.yticks([])  # remove y-axis values
    plt.savefig("./iou loss/" + test_lbl + ".png")
    plt.show()
    cnt += 1

models = [model1, model2, model3, model4, model5, model6, model7]
preprocess_functions = [preprocess_input5, preprocess_input6, preprocess_input7]


def prediction(model_idx, image, patch_size):
    
    segm_img = np.zeros(image.shape[:2])  #Array with zeros to be filled with segmented values
    patch_num=1
    for i in range(0, image.shape[0], patch_size):   #Steps of 256
        for j in range(0, image.shape[1], patch_size):  #Steps of 256
            #print(i, j)
            model = models[model_idx]
            single_patch = image[i:i+patch_size, j:j+patch_size, :]
            
            if model_idx < 4:
                single_patch_shape = single_patch.shape[:2]
                single_patch_input = np.expand_dims(single_patch, 0)
                
            if model_idx == 4:
                single_patch_input = np.expand_dims(single_patch, 0)
                preprocess_input = preprocess_functions[model_idx - 4]
                single_patch_input = preprocess_input(single_patch_input)
                single_patch_shape = single_patch.shape[:2]
                
                
            if model_idx >= 5:
                single_patch_resized = cv2.resize(single_patch, (RESIZED_SIZE, RESIZED_SIZE))
                single_patch_input = np.expand_dims(single_patch_resized, 0)
                preprocess_input = preprocess_functions[model_idx - 4]
                single_patch_input = preprocess_input(single_patch_input)
                single_patch_shape = single_patch_resized.shape[:2]
                
            #single_patch_prediction = model.predict(single_patch_input)[0,:,:,0].astype(np.uint8)
            test_pred = model.predict(single_patch_input)
            test_prediction = np.argmax(test_pred, axis=3)[0,:,:]
            test_prediction = np.where(test_prediction == 1, 0, 1)
            segm_img[i:i+single_patch_shape[0], j:j+single_patch_shape[1]] += cv2.resize(test_prediction, single_patch_shape[::-1])
          
            print("Finished processing patch number ", patch_num, " at position ", i,j)
            patch_num+=1
    return segm_img   

    
    
image_path = "./dust_1024_G05_26_27_28_29/images/"
mask_path = "./dust_1024_G05_26_27_28_29/masks/"
colored_mask_path = "./dust_1024_G05_26_27_28_29/color coded masks/"

save_path = "./dust_1024_G05_26_27_28_29/predictions/"

for img in next(os.walk(image_path))[2]:
    
    image = Image.open(image_path + img)
    mask = Image.open(mask_path + img).convert("L")
    colored_mask = Image.open(colored_mask_path + img + ".png")
    
    gt_mask = np.array(mask)
    gt_mask = gt_mask/255.
    
    
    # Resize the image while adding black pixels to the borders
    new_image = Image.new("RGB", (4096, 2048), color="black")
    new_mask = Image.new("L", (4096, 2048), color="black")
    
    new_image.paste(image, ((4096 - image.size[0]) // 2, (2048 - image.size[1]) // 2))
    new_mask.paste(mask, ((4096 - mask.size[0]) // 2, (2048 - mask.size[1]) // 2))
    
    new_image = np.array(new_image)
    new_image_resized = cv2.resize(new_image, (2048, 1024))
    
    new_mask = np.array(new_mask)
    new_mask_resized = cv2.resize(new_mask, (2048, 1024))
    
    
    # Define the crop boundaries
    left = (2048 - 1801) // 2  # Amount of padding on the left
    top = (1024 - 900) // 2  # Amount of padding on the top
    right = left + 1801  # Right boundary of the crop
    bottom = top + 900  # Bottom boundary of the crop
    
    
    seg_mask = Image.fromarray(new_mask_resized)
    seg_mask = seg_mask.crop((left, top, right, bottom))
    
    
    gt_mask_resized = np.array(seg_mask)
    gt_mask_resized = gt_mask_resized/255.
    
    
    segmented_image1 = prediction(0, new_image, 1024)
    segmented_image1 = Image.fromarray(segmented_image1)
    segmented_image1 = segmented_image1.crop((248, 124, 3848, 1925))  # left, top, right, bottom
    segmented_image1 = np.array(segmented_image1)
    segmented_image1 = np.where(segmented_image1 == 1, 0, 1)
    iou1 = calculate_iou(segmented_image1, gt_mask)
    
    
    segmented_image2 = prediction(1, new_image, 1024)
    segmented_image2 = Image.fromarray(segmented_image2)
    segmented_image2 = segmented_image2.crop((248, 124, 3848, 1925))  # left, top, right, bottom
    segmented_image2 = np.array(segmented_image2)
    segmented_image2 = np.where(segmented_image2 == 1, 0, 1)
    iou2 = calculate_iou(segmented_image2, gt_mask)
    
    
    segmented_image3 = prediction(2, new_image, 1024)
    segmented_image3 = Image.fromarray(segmented_image3)
    segmented_image3 = segmented_image3.crop((248, 124, 3848, 1925))  # left, top, right, bottom
    segmented_image3 = np.array(segmented_image3)
    segmented_image3 = np.where(segmented_image3 == 1, 0, 1)
    iou3 = calculate_iou(segmented_image3, gt_mask)
    
    
    segmented_image4 = prediction(3, new_image, 1024)
    segmented_image4 = Image.fromarray(segmented_image4)
    segmented_image4 = segmented_image4.crop((248, 124, 3848, 1925))  # left, top, right, bottom
    segmented_image4 = np.array(segmented_image4)
    segmented_image4 = np.where(segmented_image4 == 1, 0, 1)
    iou4 = calculate_iou(segmented_image4, gt_mask)
    
    
    segmented_image5 = prediction(4, new_image, 1024)
    segmented_image5 = Image.fromarray(segmented_image5)
    segmented_image5 = segmented_image5.crop((248, 124, 3848, 1925))  # left, top, right, bottom
    segmented_image5 = np.array(segmented_image5)
    segmented_image5 = np.where(segmented_image5 == 1, 0, 1)
    iou5 = calculate_iou(segmented_image5, gt_mask)
    
    
    segmented_image6 = prediction(5, new_image_resized, 512)
    segmented_image6 = Image.fromarray(segmented_image6)
    segmented_image6 = segmented_image6.crop((left, top, right, bottom))  # left, top, right, bottom
    segmented_image6 = np.array(segmented_image6)
    segmented_image6 = np.where(segmented_image6 == 1, 0, 1)
    iou6 = calculate_iou(segmented_image6, gt_mask_resized)
    
    
    segmented_image7 = prediction(6, new_image_resized, 512)
    segmented_image7 = Image.fromarray(segmented_image7)
    segmented_image7 = segmented_image7.crop((left, top, right, bottom))  # left, top, right, bottom
    segmented_image7 = np.array(segmented_image7)
    segmented_image7 = np.where(segmented_image7 == 1, 0, 1)
    iou7 = calculate_iou(segmented_image7, gt_mask_resized)
    
    
    plt.figure(figsize=(40, 5))
    plt.subplot(1, 10, 1)
    plt.title('Large Image')
    plt.imshow(image)
    plt.subplot(1, 10, 2)
    plt.title('Color Coded Mask')
    plt.imshow(colored_mask)
    plt.text(0.5, -0.15, img[:9], ha='center', va='center', transform=plt.gca().transAxes, fontsize=10)
    plt.xticks([])  # remove x-axis values
    #plt.yticks([])  # remove y-axis values
    plt.subplot(1, 10, 3)
    plt.title('Mask')
    plt.imshow(mask, cmap = 'gray')
    plt.xticks([])  # remove x-axis values
    #plt.yticks([])  # remove y-axis values
    plt.subplot(1, 10, 4)
    plt.title('unet_2d_ResNet50_1024ps')
    plt.imshow(segmented_image1)
    plt.text(0.5, -0.15, "IOU = " + str(round(iou1, 3)), ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
    plt.xticks([])  # remove x-axis values
    #plt.yticks([])  # remove y-axis values
    plt.subplot(1, 10, 5)
    plt.title('unet_2d_DenseNet121_1024ps')
    plt.imshow(segmented_image2)
    plt.text(0.5, -0.15, "IOU = " + str(round(iou2, 3)), ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
    plt.xticks([])  # remove x-axis values
    #plt.yticks([])  # remove y-axis values
    plt.subplot(1, 10, 6)
    plt.title('att_unet_2d_DenseNet201_1024ps')
    plt.imshow(segmented_image3)
    plt.text(0.5, -0.15, "IOU = " + str(round(iou3, 3)), ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
    plt.xticks([])  # remove x-axis values
    #plt.yticks([])  # remove y-axis values
    plt.subplot(1, 10, 7)
    plt.title('unet_plus_2d_ResNet152_1024ps')
    plt.imshow(segmented_image4)
    plt.text(0.5, -0.15, "IOU = " + str(round(iou4, 3)), ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
    plt.xticks([])  # remove x-axis values
    #plt.yticks([])  # remove y-axis values
    plt.subplot(1, 10, 8)
    plt.title('unet_inceptionv3_1024ps')
    plt.imshow(segmented_image5)
    plt.text(0.5, -0.15, "IOU = " + str(round(iou5, 3)), ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
    plt.xticks([])  # remove x-axis values
    #plt.yticks([])  # remove y-axis values
    plt.subplot(1, 10, 9)
    plt.title('unet_inceptionv3_512ps')
    plt.imshow(segmented_image6)
    plt.text(0.5, -0.15, "IOU = " + str(round(iou6, 3)), ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
    plt.xticks([])  # remove x-axis values
    #plt.yticks([])  # remove y-axis values
    plt.subplot(1, 10, 10)
    plt.title('unet_vgg16_512ps')
    plt.imshow(segmented_image7)
    plt.text(0.5, -0.15, "IOU = " + str(round(iou7, 3)), ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
    plt.xticks([])  # remove x-axis values
    #plt.yticks([])  # remove y-axis values
    plt.savefig(save_path + img)
    plt.show()



