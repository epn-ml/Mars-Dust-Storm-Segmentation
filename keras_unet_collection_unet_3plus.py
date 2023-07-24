# -*- coding: utf-8 -*-

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

image_directory = './attention unet/dust_1024_patch_size/img_patches/'
mask_directory = './attention unet/dust_1024_patch_size/mask_patches/'

image_dataset = [] 
mask_dataset = []

images = os.listdir(image_directory)
for i, image_name in enumerate(images):
    image = cv2.imread(image_directory+image_name)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image_dataset.append(np.array(image))

masks = os.listdir(mask_directory)
for i, image_name in enumerate(masks):
    image = cv2.imread(mask_directory+image_name, 0)
    image = Image.fromarray(image)
    mask_dataset.append(np.array(image))

image_dataset = np.array(image_dataset)
mask_dataset = np.array(mask_dataset)

train_images = image_dataset
train_masks = mask_dataset


#Sanity check, view few mages
import random
import numpy as np
image_number = random.randint(0, train_images.shape[0])
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(train_images[image_number])
plt.subplot(122)
plt.imshow(train_masks[image_number], cmap='gray')
plt.show()

#Encode labels... but multi dim array so need to flatten, encode and reshape
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
n, h, w = train_masks.shape
train_masks_reshaped = train_masks.reshape(-1,1)
train_masks_reshaped_encoded = labelencoder.fit_transform(train_masks_reshaped)
train_masks_encoded_original_shape = train_masks_reshaped_encoded.reshape(n, h, w)

np.unique(train_masks_encoded_original_shape)
#################################################
train_masks_input = np.expand_dims(train_masks_encoded_original_shape, axis=3)

#Create a subset of data for quick testing
#Picking 10% for testing and remaining for training
from sklearn.model_selection import train_test_split
X1, X_test, y1, y_test = train_test_split(train_images, train_masks_input, test_size = 0.2, random_state = 0)

X_train, X_do_not_use, y_train, y_do_not_use = X1, X_test, y1, y_test
print("Class values in the dataset are ... ", np.unique(y_train))  # 0 is the background


n_classes=2
from keras.utils import to_categorical
train_masks_cat = to_categorical(y_train, num_classes=n_classes)
y_train_cat = train_masks_cat.reshape((y_train.shape[0], y_train.shape[1], y_train.shape[2], n_classes))


test_masks_cat = to_categorical(y_test, num_classes=n_classes)
y_test_cat = test_masks_cat.reshape((y_test.shape[0], y_test.shape[1], y_test.shape[2], n_classes))

# save to csv file
save('./X_train.npy', X_train)
save('./X_test.npy', X_test)
save('./y_train_cat.npy', y_train_cat)
save('./y_test_cat.npy', y_test_cat)



X_train = load('./X_train.npy')
X_test = load('./X_test.npy')
y_train_cat = load('./y_train_cat.npy')
y_test_cat = load('./y_test_cat.npy')
#######################################

IMG_HEIGHT = X_train.shape[1]
IMG_WIDTH  = X_train.shape[2]
IMG_CHANNELS = X_train.shape[3]
num_labels = 2  #Binary
input_shape = (IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS)
batch_size = 1
###############################################################################
from keras_unet_collection import models, losses

def iou_score(y_true, y_pred):
    # convert y_pred to binary mask
    y_pred = tf.cast(y_pred > 0.5, dtype=tf.float32)
    # calculate intersection and union
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2])
    union = tf.reduce_sum(y_true + y_pred, axis=[1, 2]) - intersection
    # calculate iou score
    iou = intersection / union
    # return mean iou score across batch
    return tf.reduce_mean(iou)
###############################################################################
model_Unet = models.unet_3plus_2d((IMG_HEIGHT, IMG_WIDTH, 3), filter_num_down=[16, 32, 64, 128], 
                           n_labels=num_labels, 
                           stack_num_down=2, stack_num_up=2, 
                           activation='ReLU', 
                           output_activation='Sigmoid', 
                           batch_norm=True, pool=False, unpool=False, 
                           backbone='ResNet50V2', weights='imagenet', 
                           freeze_backbone=True, freeze_batch_norm=True)

model_Unet.compile(loss='binary_crossentropy', optimizer=Adam(lr = 1e-3), metrics=[iou_score])

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='./unet 3plus/1024 patch size/unet_3plus_2d_ResNet50V2_100epochs_G05_1024patchsize.h5',
    monitor='val_iou_score',
    mode='max',
    save_best_only=True, verbose=1)

Unet_history = model_Unet.fit(X_train, 
                   y_train_cat, 
                    verbose=1,
                    batch_size = batch_size,
                    validation_data=(X_test, y_test_cat), 
                    epochs=100,
                    callbacks=[model_checkpoint_callback])

#plot the training and validation accuracy and loss at each epoch
loss = Unet_history.history['loss']
val_loss = Unet_history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss unet_3plus_2d_ResNet50V2_100epochs_G05_1024patchsize')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()


acc = Unet_history.history['iou_score']
val_acc = Unet_history.history['val_iou_score']

plt.plot(epochs, acc, 'y', label='Training IOU')
plt.plot(epochs, val_acc, 'r', label='Validation IOU')
plt.title('Training and validation IOU unet_3plus_2d_ResNet50V2_100epochs_G05_1024patchsize')
plt.xlabel('Epochs')
plt.ylabel('IOU')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()


import csv


rows = zip(loss, val_loss, acc, val_acc)
with open('./unet 3plus//1024 patch size/unet_3plus_2d_ResNet50V2.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Training loss', 'Validation loss', 'Training IOU', 'Validation IOU'])
    for row in rows:
        writer.writerow(row)
    
#############################################################
## The same for the backbones mentioned in the thesis
#######################################################



from keras.models import load_model

#Set compile=False as we are not loading it for training, only for prediction.
model1 = load_model('./unet 3plus/1024 patch size/models/unet_3plus_2d/unet_3plus_2d_DenseNet121_100epochs_G05_1024patchsize.h5', compile=False)
model2 = load_model('./unet 3plus/1024 patch size/models/unet_3plus_2d/unet_3plus_2d_DenseNet169_100epochs_G05_1024patchsize.h5', compile=False)
model3 = load_model('./unet 3plus/1024 patch size/models/unet_3plus_2d/unet_3plus_2d_DenseNet201_100epochs_G05_1024patchsize.h5', compile=False)
model4 = load_model('./unet 3plus/1024 patch size/models/unet_3plus_2d/unet_3plus_2d_ResNet50_100epochs_G05_1024patchsize.h5', compile=False)
model5 = load_model('./unet 3plus/1024 patch size/models/unet_3plus_2d/unet_3plus_2d_ResNet50V2_100epochs_G05_1024patchsize.h5', compile=False)
model6 = load_model('./unet 3plus/1024 patch size/models/unet_3plus_2d/unet_3plus_2d_ResNet101_100epochs_G05_1024patchsize.h5', compile=False)
model7 = load_model('./unet 3plus/1024 patch size/models/unet_3plus_2d/unet_3plus_2d_ResNet101V2_100epochs_G05_1024patchsize.h5', compile=False)
model8 = load_model('./unet 3plus/1024 patch size/models/unet_3plus_2d/unet_3plus_2d_resnet152_100epochs_G05_1024patchsize.h5', compile=False)
model9 = load_model('./unet 3plus/1024 patch size/models/unet_3plus_2d/unet_3plus_2d_resnet152v2_100epochs_G05_1024patchsize.h5', compile=False)
model10 = load_model('./unet 3plus/1024 patch size/models/unet_3plus_2d/unet_3plus_2d_VGG16_100epochs_G05_1024patchsize.h5', compile=False)
model11 = load_model('./unet 3plus/1024 patch size/models/unet_3plus_2d/unet_3plus_2d_VGG19_100epochs_G05_1024patchsize.h5', compile=False)

##############################################################
#Test some random images
import random

for test_img_number in range(len(X_test)):
    test_img = X_test[test_img_number]
    ground_truth=y_test[test_img_number]
    test_img_input=np.expand_dims(test_img, 0)

    test_pred1 = model1.predict(test_img_input)
    test_prediction1 = np.argmax(test_pred1, axis=3)[0,:,:]

    test_pred2 = model2.predict(test_img_input)
    test_prediction2 = np.argmax(test_pred2, axis=3)[0,:,:]

    test_pred3 = model3.predict(test_img_input)
    test_prediction3 = np.argmax(test_pred3, axis=3)[0,:,:]

    test_pred4 = model4.predict(test_img_input)
    test_prediction4 = np.argmax(test_pred4, axis=3)[0,:,:]

    test_pred5 = model5.predict(test_img_input)
    test_prediction5 = np.argmax(test_pred5, axis=3)[0,:,:]

    test_pred6 = model6.predict(test_img_input)
    test_prediction6 = np.argmax(test_pred6, axis=3)[0,:,:]

    test_pred7 = model7.predict(test_img_input)
    test_prediction7 = np.argmax(test_pred7, axis=3)[0,:,:]

    test_pred8 = model8.predict(test_img_input)
    test_prediction8 = np.argmax(test_pred8, axis=3)[0,:,:]

    test_pred9 = model9.predict(test_img_input)
    test_prediction9 = np.argmax(test_pred9, axis=3)[0,:,:]

    test_pred10 = model10.predict(test_img_input)
    test_prediction10 = np.argmax(test_pred10, axis=3)[0,:,:]

    test_pred11 = model11.predict(test_img_input)
    test_prediction11 = np.argmax(test_pred11, axis=3)[0,:,:]


    p = X_test[test_img_number,:,:,:].squeeze()

    plt.figure(figsize=(22, 5))
    plt.subplot(1, 13, 1)
    plt.title('Testing Image')
    plt.imshow(p)
    plt.subplot(1, 13, 2)
    plt.title('Testing Label')
    plt.imshow(ground_truth[:,:,0], cmap='gray')
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    plt.subplot(1, 13, 3)
    plt.title('densenet121')
    plt.imshow(test_prediction1)
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    plt.subplot(1, 13, 4)
    plt.title('densenet169')
    plt.imshow(test_prediction2)
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    plt.subplot(1, 13, 5)
    plt.title('densenet201')
    plt.imshow(test_prediction3)
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    plt.subplot(1, 13, 6)
    plt.title('resnet50')
    plt.imshow(test_prediction4)
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    plt.subplot(1, 13, 7)
    plt.title('resnet50v2')
    plt.imshow(test_prediction5)
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    plt.subplot(1, 13, 8)
    plt.title('resnet101')
    plt.imshow(test_prediction6)
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    plt.subplot(1, 13, 9)
    plt.title('resnet101v2')
    plt.imshow(test_prediction7)
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    plt.subplot(1, 13, 10)
    plt.title('resnet152')
    plt.imshow(test_prediction8)
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    plt.subplot(1, 13, 11)
    plt.title('resnet152v2')
    plt.imshow(test_prediction9)
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    plt.subplot(1, 13, 12)
    plt.title('vgg16')
    plt.imshow(test_prediction10)
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    plt.subplot(1, 13, 13)
    plt.title('vgg19')
    plt.imshow(test_prediction11)
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    if test_img_number < 10:
        plt.savefig("./unet 3plus/1024 patch size/models/unet_3plus_2d/prediction of val images/" + "0" + str(test_img_number) + ".png")
    else:
        plt.savefig("./unet 3plus/1024 patch size/models/unet_3plus_2d/prediction of val images/" + str(test_img_number) + ".png")
 
    plt.show()    


test_labels = ["25_03", "25_10", "25_11", "25_12", "25_13", "26_00", "26_01", 
               "26_02", "26_03", "26_11", "26_12", "27_00", "27_01", "27_02", 
               "27_03", "27_10", "27_11", "28_00", "28_01", "28_02", "28_03", 
               "28_10", "28_11", "29_00", "29_01", "29_02", "29_03", "29_10", 
               "30_00", "30_01"]

for test_lbl in test_labels:
    test_img = cv2.imread("./dust_1024_patch_size/test_img/image_G05_day" 
                          + test_lbl + ".png")
    test_img = cv2.cvtColor(test_img,cv2.COLOR_BGR2RGB)

    ground_truth=cv2.imread("./dust_1024_patch_size/test_mask/image_G05_day" 
                          + test_lbl + ".png", 0)

    test_img=np.expand_dims(test_img, 0)


    test_pred1 = model1.predict(test_img)
    test_prediction1 = np.argmax(test_pred1, axis=3)[0,:,:]

    test_pred2 = model2.predict(test_img)
    test_prediction2 = np.argmax(test_pred2, axis=3)[0,:,:]

    test_pred3 = model3.predict(test_img)
    test_prediction3 = np.argmax(test_pred3, axis=3)[0,:,:]

    test_pred4 = model4.predict(test_img)
    test_prediction4 = np.argmax(test_pred4, axis=3)[0,:,:]

    test_pred5 = model5.predict(test_img)
    test_prediction5 = np.argmax(test_pred5, axis=3)[0,:,:]

    test_pred6 = model6.predict(test_img)
    test_prediction6 = np.argmax(test_pred6, axis=3)[0,:,:]

    test_pred7 = model7.predict(test_img)
    test_prediction7 = np.argmax(test_pred7, axis=3)[0,:,:]

    test_pred8 = model8.predict(test_img)
    test_prediction8 = np.argmax(test_pred8, axis=3)[0,:,:]

    test_pred9 = model9.predict(test_img)
    test_prediction9 = np.argmax(test_pred9, axis=3)[0,:,:]

    test_pred10 = model10.predict(test_img)
    test_prediction10 = np.argmax(test_pred10, axis=3)[0,:,:]

    test_pred11 = model11.predict(test_img)
    test_prediction11 = np.argmax(test_pred11, axis=3)[0,:,:]


    plt.figure(figsize=(22, 5))
    plt.subplot(1, 13, 1)
    plt.title('Testing Image')
    plt.imshow(test_img.squeeze())
    plt.subplot(1, 13, 2)
    plt.title('Testing Label')
    plt.imshow(ground_truth, cmap='gray')
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    plt.subplot(1, 13, 3)
    plt.title('densenet121')
    plt.imshow(test_prediction1)
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    plt.subplot(1, 13, 4)
    plt.title('densenet169')
    plt.imshow(test_prediction2)
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    plt.subplot(1, 13, 5)
    plt.title('densenet201')
    plt.imshow(test_prediction3)
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    plt.subplot(1, 13, 6)
    plt.title('resnet50')
    plt.imshow(test_prediction4)
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    plt.subplot(1, 13, 7)
    plt.title('resnet50v2')
    plt.imshow(test_prediction5)
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    plt.subplot(1, 13, 8)
    plt.title('resnet101')
    plt.imshow(test_prediction6)
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    plt.subplot(1, 13, 9)
    plt.title('resnet101v2')
    plt.imshow(test_prediction7)
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    plt.subplot(1, 13, 10)
    plt.title('resnet152')
    plt.imshow(test_prediction8)
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    plt.subplot(1, 13, 11)
    plt.title('resnet152v2')
    plt.imshow(test_prediction9)
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    plt.subplot(1, 13, 12)
    plt.title('vgg16')
    plt.imshow(test_prediction10)
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    plt.subplot(1, 13, 13)
    plt.title('vgg19')
    plt.imshow(test_prediction11)
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    plt.savefig("./unet 3plus/1024 patch size/models/unet_3plus_2d/prediction of test images/" + test_lbl + ".png")
 
    plt.show()


test_image_path = './dust_1024_patch_size/test_img/'
test_mask_path = './dust_1024_patch_size/test_mask/'

test_image_dataset = []
test_mask_dataset = [] 

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
    #image = image.resize((SIZE, SIZE))
    test_mask_dataset.append(np.array(image))
    

test_image_dataset = np.array(test_image_dataset)
test_mask_dataset = np.array(test_mask_dataset)

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
models = [model1, model2, model3, model4, model5, model6, model7, model8, model9, model10, model11]

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
        
        test_pred = model.predict(test_img_input)
        test_prediction = np.argmax(test_pred, axis=3)[0,:,:]
        
        
        gt_mask = test_mask_dataset[i]
        gt_mask = gt_mask/255.
        
        # Calculate the IOU for this test image
        iou = calculate_iou(test_prediction, gt_mask)
        iou_values.append(iou)
    
    # add a new column to the dataframe for this model's IOU values
    iou_df[f'model{model_idx+1}'] = iou_values
  
    
# create a new dataframe to store the average IOU values for each model
avg_iou_df = pd.DataFrame(columns=[f'model{i}' for i in range(1, 12)])

# compute the average IOU values for each model and add them to the new dataframe
for model_idx, model in enumerate(models):
    avg_iou = sum(iou_df[f'model{model_idx+1}']) / len(iou_df[f'model{model_idx+1}'])
    avg_iou_df.loc[0, f'model{model_idx+1}'] = avg_iou
    
    
# Save the dataframe to a CSV file on your computer
iou_df.to_csv('./unet 3plus/1024 patch size/models/unet_3plus_2d/iou_values_for_test_images.csv', index=False)
avg_iou_df.to_csv('./unet 3plus/1024 patch size/models/unet_3plus_2d/avg_iou_values_for_test_images.csv', index=False)



cnt = 0

test_labels = ["25_03", "25_10", "25_11", "25_12", "25_13", "26_00", "26_01", 
               "26_02", "26_03", "26_11", "26_12", "27_00", "27_01", "27_02", 
               "27_03", "27_10", "27_11", "28_00", "28_01", "28_02", "28_03", 
               "28_10", "28_11", "29_00", "29_01", "29_02", "29_03", "29_10", 
               "30_00", "30_01"]

for test_lbl in test_labels:
    test_img = cv2.imread("./dust_1024_patch_size/test_img/image_G05_day" 
                          + test_lbl + ".png")
    test_img = cv2.cvtColor(test_img,cv2.COLOR_BGR2RGB)

    ground_truth=cv2.imread("./dust_1024_patch_size/test_mask/image_G05_day" 
                          + test_lbl + ".png", 0)

    test_img=np.expand_dims(test_img, 0)


    test_pred1 = model1.predict(test_img)
    test_prediction1 = np.argmax(test_pred1, axis=3)[0,:,:]

    test_pred2 = model2.predict(test_img)
    test_prediction2 = np.argmax(test_pred2, axis=3)[0,:,:]

    test_pred3 = model3.predict(test_img)
    test_prediction3 = np.argmax(test_pred3, axis=3)[0,:,:]

    test_pred4 = model4.predict(test_img)
    test_prediction4 = np.argmax(test_pred4, axis=3)[0,:,:]

    test_pred5 = model5.predict(test_img)
    test_prediction5 = np.argmax(test_pred5, axis=3)[0,:,:]

    test_pred6 = model6.predict(test_img)
    test_prediction6 = np.argmax(test_pred6, axis=3)[0,:,:]

    test_pred7 = model7.predict(test_img)
    test_prediction7 = np.argmax(test_pred7, axis=3)[0,:,:]

    test_pred8 = model8.predict(test_img)
    test_prediction8 = np.argmax(test_pred8, axis=3)[0,:,:]

    test_pred9 = model9.predict(test_img)
    test_prediction9 = np.argmax(test_pred9, axis=3)[0,:,:]

    test_pred10 = model10.predict(test_img)
    test_prediction10 = np.argmax(test_pred10, axis=3)[0,:,:]

    test_pred11 = model11.predict(test_img)
    test_prediction11 = np.argmax(test_pred11, axis=3)[0,:,:]


    plt.figure(figsize=(22, 5))
    plt.subplot(1, 13, 1)
    plt.title('Testing Image')
    plt.imshow(test_img.squeeze())
    plt.subplot(1, 13, 2)
    plt.title('Testing Label')
    plt.imshow(ground_truth, cmap='gray')
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    plt.subplot(1, 13, 3)
    plt.title('densenet121')
    plt.imshow(test_prediction1)
    plt.text(0.5, -0.1, str(round(iou_df["model1"][cnt], 3)), ha='center', va='center', transform=plt.gca().transAxes, fontsize=10)
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    plt.subplot(1, 13, 4)
    plt.title('densenet169')
    plt.imshow(test_prediction2)
    plt.text(0.5, -0.1, str(round(iou_df["model2"][cnt], 3)), ha='center', va='center', transform=plt.gca().transAxes, fontsize=10)
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    plt.subplot(1, 13, 5)
    plt.title('densenet201')
    plt.imshow(test_prediction3)
    plt.text(0.5, -0.1, str(round(iou_df["model3"][cnt], 3)), ha='center', va='center', transform=plt.gca().transAxes, fontsize=10)
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    plt.subplot(1, 13, 6)
    plt.title('resnet50')
    plt.imshow(test_prediction4)
    plt.text(0.5, -0.1, str(round(iou_df["model4"][cnt], 3)), ha='center', va='center', transform=plt.gca().transAxes, fontsize=10)
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    plt.subplot(1, 13, 7)
    plt.title('resnet50v2')
    plt.imshow(test_prediction5)
    plt.text(0.5, -0.1, str(round(iou_df["model5"][cnt], 3)), ha='center', va='center', transform=plt.gca().transAxes, fontsize=10)
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    plt.subplot(1, 13, 8)
    plt.title('resnet101')
    plt.text(0.5, -0.1, str(round(iou_df["model6"][cnt], 3)), ha='center', va='center', transform=plt.gca().transAxes, fontsize=10)
    plt.imshow(test_prediction6)
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    plt.subplot(1, 13, 9)
    plt.title('resnet101v2')
    plt.imshow(test_prediction7)
    plt.text(0.5, -0.1, str(round(iou_df["model7"][cnt], 3)), ha='center', va='center', transform=plt.gca().transAxes, fontsize=10)
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    plt.subplot(1, 13, 10)
    plt.title('resnet152')
    plt.imshow(test_prediction8)
    plt.text(0.5, -0.1, str(round(iou_df["model8"][cnt], 3)), ha='center', va='center', transform=plt.gca().transAxes, fontsize=10)
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    plt.subplot(1, 13, 11)
    plt.title('resnet152v2')
    plt.imshow(test_prediction9)
    plt.text(0.5, -0.1, str(round(iou_df["model9"][cnt], 3)), ha='center', va='center', transform=plt.gca().transAxes, fontsize=10)
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    plt.subplot(1, 13, 12)
    plt.title('vgg16')
    plt.imshow(test_prediction10)
    plt.text(0.5, -0.1, str(round(iou_df["model10"][cnt], 3)), ha='center', va='center', transform=plt.gca().transAxes, fontsize=10)
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    plt.subplot(1, 13, 13)
    plt.title('vgg19')
    plt.imshow(test_prediction11)
    plt.text(0.5, -0.1, str(round(iou_df["model11"][cnt], 3)), ha='center', va='center', transform=plt.gca().transAxes, fontsize=10)
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    plt.savefig("./unet 3plus/1024 patch size/models/unet_3plus_2d/prediction of test images with IOU/" + test_lbl + ".png")
    plt.show()
    cnt += 1


for test_img_number in range(len(X_test)):
    test_img = X_test[test_img_number]
    ground_truth=y_test[test_img_number]
    gt_mask = ground_truth
    gt_mask = ground_truth/255.
    gt_mask = gt_mask.squeeze()
    test_img_input=np.expand_dims(test_img, 0)

    test_pred1 = model1.predict(test_img_input)
    test_prediction1 = np.argmax(test_pred1, axis=3)[0,:,:]
    iou1 = calculate_iou(test_prediction1, gt_mask)

    test_pred2 = model2.predict(test_img_input)
    test_prediction2 = np.argmax(test_pred2, axis=3)[0,:,:]
    iou2 = calculate_iou(test_prediction2, gt_mask)

    test_pred3 = model3.predict(test_img_input)
    test_prediction3 = np.argmax(test_pred3, axis=3)[0,:,:]
    iou3 = calculate_iou(test_prediction3, gt_mask)

    test_pred4 = model4.predict(test_img_input)
    test_prediction4 = np.argmax(test_pred4, axis=3)[0,:,:]
    iou4 = calculate_iou(test_prediction4, gt_mask)

    test_pred5 = model5.predict(test_img_input)
    test_prediction5 = np.argmax(test_pred5, axis=3)[0,:,:]
    iou5 = calculate_iou(test_prediction5, gt_mask)

    test_pred6 = model6.predict(test_img_input)
    test_prediction6 = np.argmax(test_pred6, axis=3)[0,:,:]
    iou6 = calculate_iou(test_prediction6, gt_mask)

    test_pred7 = model7.predict(test_img_input)
    test_prediction7 = np.argmax(test_pred7, axis=3)[0,:,:]
    iou7 = calculate_iou(test_prediction7, gt_mask)
    
    test_pred8 = model8.predict(test_img_input)
    test_prediction8 = np.argmax(test_pred8, axis=3)[0,:,:]
    iou8 = calculate_iou(test_prediction8, gt_mask)

    test_pred9 = model9.predict(test_img_input)
    test_prediction9 = np.argmax(test_pred9, axis=3)[0,:,:]
    iou9 = calculate_iou(test_prediction9, gt_mask)

    test_pred10 = model10.predict(test_img_input)
    test_prediction10 = np.argmax(test_pred10, axis=3)[0,:,:]
    iou10 = calculate_iou(test_prediction10, gt_mask)

    test_pred11 = model11.predict(test_img_input)
    test_prediction11 = np.argmax(test_pred11, axis=3)[0,:,:]
    iou11 = calculate_iou(test_prediction11, gt_mask)


    p = X_test[test_img_number,:,:,:].squeeze()

    plt.figure(figsize=(22, 5))
    plt.subplot(1, 13, 1)
    plt.title('Testing Image')
    plt.imshow(p)
    plt.subplot(1, 13, 2)
    plt.title('Testing Label')
    plt.imshow(ground_truth[:,:,0], cmap='gray')
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    plt.subplot(1, 13, 3)
    plt.title('densenet121')
    plt.imshow(test_prediction1)
    plt.text(0.5, -0.1, str(round(iou1, 3)), ha='center', va='center', transform=plt.gca().transAxes, fontsize=10)
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    plt.subplot(1, 13, 4)
    plt.title('densenet169')
    plt.imshow(test_prediction2)
    plt.text(0.5, -0.1, str(round(iou2, 3)), ha='center', va='center', transform=plt.gca().transAxes, fontsize=10)
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    plt.subplot(1, 13, 5)
    plt.title('densenet201')
    plt.imshow(test_prediction3)
    plt.text(0.5, -0.1, str(round(iou3, 3)), ha='center', va='center', transform=plt.gca().transAxes, fontsize=10)
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    plt.subplot(1, 13, 6)
    plt.title('resnet50')
    plt.imshow(test_prediction4)
    plt.text(0.5, -0.1, str(round(iou4, 3)), ha='center', va='center', transform=plt.gca().transAxes, fontsize=10)
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    plt.subplot(1, 13, 7)
    plt.title('resnet50v2')
    plt.imshow(test_prediction5)
    plt.text(0.5, -0.1, str(round(iou5, 3)), ha='center', va='center', transform=plt.gca().transAxes, fontsize=10)
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    plt.subplot(1, 13, 8)
    plt.title('resnet101')
    plt.imshow(test_prediction6)
    plt.text(0.5, -0.1, str(round(iou6, 3)), ha='center', va='center', transform=plt.gca().transAxes, fontsize=10)
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    plt.subplot(1, 13, 9)
    plt.title('resnet101v2')
    plt.imshow(test_prediction7)
    plt.text(0.5, -0.1, str(round(iou7, 3)), ha='center', va='center', transform=plt.gca().transAxes, fontsize=10)
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    plt.subplot(1, 13, 10)
    plt.title('resnet152')
    plt.imshow(test_prediction8)
    plt.text(0.5, -0.1, str(round(iou8, 3)), ha='center', va='center', transform=plt.gca().transAxes, fontsize=10)
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    plt.subplot(1, 13, 11)
    plt.title('resnet152v2')
    plt.imshow(test_prediction9)
    plt.text(0.5, -0.1, str(round(iou9, 3)), ha='center', va='center', transform=plt.gca().transAxes, fontsize=10)
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    plt.subplot(1, 13, 12)
    plt.title('vgg16')
    plt.imshow(test_prediction10)
    plt.text(0.5, -0.1, str(round(iou10, 3)), ha='center', va='center', transform=plt.gca().transAxes, fontsize=10)
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    plt.subplot(1, 13, 13)
    plt.title('vgg19')
    plt.imshow(test_prediction11)
    plt.text(0.5, -0.1, str(round(iou11, 3)), ha='center', va='center', transform=plt.gca().transAxes, fontsize=10)
    plt.xticks([])  # remove x-axis values
    plt.yticks([])  # remove y-axis values
    if test_img_number < 10:
        plt.savefig("./unet 3plus/1024 patch size/models/unet_3plus_2d/prediction of val images with IOU/" + "0" + str(test_img_number) + ".png")
    else:
        plt.savefig("./unet 3plus/1024 patch size/models/unet_3plus_2d/prediction of val images with IOU/" + str(test_img_number) + ".png")
 
    plt.show()
