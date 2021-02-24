import os
import glob
import random 
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn 
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import helpers
import unet

random.seed(42) # set seed for suffling of data
IMG_DIM = 192
BATCH_SIZE = 32
NUM_CLASSES = 4

parent_path = os.path.dirname( os.getcwd() )
data_path = os.path.join(parent_path, "labelled/")

files_gt = glob.glob(os.path.join(data_path, '*_gt.nii.gz'))
files_sa = glob.glob(os.path.join(data_path, '*_sa.nii.gz'))
nr_patients = len(list( files_gt ) ) 

# store nr of slices for each patient in list
nr_slices_list = []
for i, current_file in enumerate(files_gt):
    patient_img = nib.load(current_file).get_fdata()
    nr_slices_list.append(patient_img.shape[2])

# shuffle images and split into training/validation/test set
c = list(zip(files_sa, files_gt, nr_slices_list))
random.shuffle(c)
files_sa_shuffled, files_gt_shuffled, nr_slices_list_shuffled = zip(*c)

files_sa_train, files_sa_validate, files_sa_test = np.split(files_sa_shuffled, [int(.7*len(files_sa_shuffled)), int(.8*len(files_sa_shuffled)) ] ) 
files_gt_train, files_gt_validate, files_gt_test = np.split(files_gt_shuffled, [int(.7*len(files_gt_shuffled)), int(.8*len(files_gt_shuffled)) ] ) 
nr_slices_train, nr_slices_validate, nr_slices_test = np.split(nr_slices_list_shuffled, [int(.7*len(nr_slices_list_shuffled)), int(.8*len(nr_slices_list)) ] ) 

test_sa = helpers.load_images(files_sa_test, IMG_DIM, 2*sum(nr_slices_test))
test_sa = helpers.preprocess_test(test_sa)
test_sa = test_sa[np.newaxis,...]
test_sa = np.transpose(test_sa, (3, 0, 1, 2)) # rearrange axes for testing

test_gt = helpers.load_images(files_gt_test, IMG_DIM, 2*sum(nr_slices_test))
test_gt = test_gt[np.newaxis,...]
test_gt = np.transpose(test_gt, (3, 0, 1, 2)) # rearrange axes for testing

# add test data to pytorch data set
test_dataset = TensorDataset( torch.Tensor(test_sa), torch.Tensor(test_gt) )
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# load model
model = unet.UNet(num_classes=NUM_CLASSES) # create instance of unet class
model.load_state_dict(torch.load('unet_conv_net_model3_550.ckpt')) # load weights of pretrained model
model.eval()
dice_score = np.zeros(test_sa.shape[0])

with torch.no_grad():
    for i, (image, label) in enumerate(test_dataloader):
        prediction = model(image) 
        segmented_img = torch.argmax(prediction,1) # for each pixel choose class with highest probability
        label_img = torch.squeeze(label)

        # dice score between the predicted image and label
        for j in range(image.shape[0]):
            curr_img = segmented_img[j,:,:]
            curr_lab = label_img[j,:,:]
            dice_score[i*BATCH_SIZE + j] = helpers.dice_metric(curr_lab,curr_img,NUM_CLASSES)

        # plot some examples: original image, label and image prediction
        print(image[3,:,:].shape)
        f, axsarr = plt.subplots(3,3,figsize=(9,9))
        axsarr[0,0].imshow(torch.squeeze(image[3,:,:]), cmap='gray')
        axsarr[0,0].set_title('original image')
        axsarr[0,1].imshow(label_img[3,:,:])
        axsarr[0,1].set_title('label')
        axsarr[0,2].imshow(segmented_img[3,:,:])
        axsarr[0,2].set_title('prediction')

        axsarr[1,0].imshow(torch.squeeze(image[26,:,:]), cmap='gray')
        axsarr[1,1].imshow(label_img[26,:,:])
        axsarr[1,2].imshow(segmented_img[26,:,:])

        axsarr[2,0].imshow(torch.squeeze(image[6,:,:]), cmap='gray')
        axsarr[2,1].imshow(label_img[6,:,:])
        axsarr[2,2].imshow(segmented_img[6,:,:])
        
        axsarr[0,0].axis('off') 
        axsarr[0,1].axis('off') 
        axsarr[0,2].axis('off') 
        axsarr[1,0].axis('off') 
        axsarr[1,1].axis('off') 
        axsarr[1,2].axis('off') 
        axsarr[2,0].axis('off') 
        axsarr[2,1].axis('off') 
        axsarr[2,2].axis('off') 

        plt.show()

# plot boxplot of dice score  
fig2, ax2 = plt.subplots()
ax2.set_title('Boxplot dice score')
ax2.boxplot(dice_score)
plt.show()
print( np.mean(dice_score), np.std(dice_score))



