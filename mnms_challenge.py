
import os
import glob
import random 
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
import gryds
import torch
import torch.nn as nn 
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import unet
import helpers

# hyperparameters
num_epochs = 4
batch_size = 32
learning_rate = 0.0001
IMG_DIM = 192

random.seed(42)

parent_path = os.path.dirname( os.getcwd() )
data_path = os.path.join(parent_path, "labelled/")

files_gt = glob.glob(os.path.join(data_path, '*_gt.nii.gz'))
files_sa = glob.glob(os.path.join(data_path, '*_sa.nii.gz'))
nr_patients = len(list( files_gt ) ) 

nr_slices_list = []
for i, current_file in enumerate(files_gt):
    patient_img = nib.load(current_file).get_fdata()
    nr_slices_list.append(patient_img.shape[2])

# ======================================================================================
# ==================| split data into training/validation/test | =======================
# ======================================================================================

c = list(zip(files_sa, files_gt, nr_slices_list))
random.shuffle(c)
files_sa_shuffled, files_gt_shuffled, nr_slices_list_shuffled = zip(*c)

files_sa_train, files_sa_validate, files_sa_test = np.split(files_sa_shuffled, [int(.7*len(files_sa_shuffled)), int(.8*len(files_sa_shuffled)) ] ) 
files_gt_train, files_gt_validate, files_gt_test = np.split(files_gt_shuffled, [int(.7*len(files_gt_shuffled)), int(.8*len(files_gt_shuffled)) ] ) 
nr_slices_train, nr_slices_validate, nr_slices_test = np.split(nr_slices_list_shuffled, [int(.7*len(nr_slices_list_shuffled)), int(.8*len(nr_slices_list)) ] ) 

train_sa = helpers.load_images(files_sa_train, IMG_DIM, 2*sum(nr_slices_train))
validate_sa = helpers.load_images(files_sa_validate, IMG_DIM, 2*sum(nr_slices_validate))
test_sa = helpers.load_images(files_sa_test, IMG_DIM, 2*sum(nr_slices_test))

train_gt = helpers.load_images(files_gt_train, IMG_DIM, 2*sum(nr_slices_train))
validate_gt = helpers.load_images(files_gt_validate, IMG_DIM, 2*sum(nr_slices_validate))
test_gt = helpers.load_images(files_gt_test, IMG_DIM, 2*sum(nr_slices_test))

train_sa_augmented, train_gt_augmented = helpers.data_augmentation(train_sa, train_gt, 10)

train_dataset = TensorDataset( torch.Tensor(train_sa_augmented), torch.Tensor(train_gt_augmented) )
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# plt.subplot(121)
# plt.imshow(train_sa_augmented[:,:,1])
# plt.subplot(122)
# plt.imshow(train_gt_augmented[:,:,1])
# plt.show()
      
        
# # # # ======================================================================================
# # # # =============================| train network | =======================================
# # # # ======================================================================================

unet_model = unet.UNet(num_classes=4)
loss_function = nn.CrossEntropyLoss() 
optimizer = optim.Adam(unet_model.parameters(), lr=learning_rate)

running_loss = 0 
printfreq = 1
savefreq = 2
for epoch in range(num_epochs):
    for i, data in enumerate(train_dataloader):
        inputs, labels = data
        optimizer.zero_grad()
        outputs =unet_model(inputs) # forward prop
        labels = labels.type(torch.LongTensor)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step() 

        running_loss += loss.item()
        if i % printfreq == printfreq-1:  
            print(epoch, i+1, running_loss / printfreq, flush=True)
            running_loss = 0 
    
    # save model 
    # if epoch % printfreq == 0:
    torch.save(unet_model.state_dict(),  'unet_conv_net_model'+str(epoch)+'.ckpt')


# # load model
# model = unet.UNet(num_classes=4)
# model.load_state_dict(torch.load('unet_conv_net_model.ckpt'))
# model.eval()