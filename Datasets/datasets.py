# test dataloader with my images. 

import torch
import torch.nn as nn
from torch.utils import data

import os
import random
import math
import gc

import numpy as np
import cv2 

class Dataset_unet_train(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, list_IDs, transform = None):
        'Initialization'
        self.list_IDs = list_IDs
        self.transform = transform

        self.img_path = 'images/datasets_2/train/img/'
        self.msk_path = 'images/datasets_2/train/msk/'

        self.fg_files = os.listdir(self.img_path)
        
    def label_mask(self, mask):
        newMask_1 = np.zeros(mask.shape[:2])
        newMask_1[mask[:,:,0] == 163] = 1

        newMask_2 = np.zeros(mask.shape[:2])
        newMask_2[mask[:,:,0] == 28] = 1

        newMask_3 = np.zeros(mask.shape[:2])
        newMask_3[mask[:,:,0] == 110] = 1

        newMask_4 = np.zeros(mask.shape[:2])
        newMask_4[mask[:,:,0] == 39] = 1

        mask = np.asarray([newMask_1, newMask_2, newMask_3, newMask_4]).astype(np.float32)

        return mask
    
    def cropped(self, index):
        crop_width = 224
        crop_height = 224 
        
        image = cv2.imread('AIRA_Images/Images/' + str(index) + '.jpg')
        mask = cv2.imread('AIRA_Images/Images/' + str(index) + '_label.png')
        
        max_x = image.shape[1] - crop_width
        max_y = image.shape[0] - crop_height

        x = np.random.randint(0, max_x)
        y = np.random.randint(0, max_y)

        crop_image = image[y: y + crop_height, x: x + crop_width]
        crop_mask  = mask[y: y + crop_height, x: x + crop_width]
        
        return crop_image, crop_mask
        
    def __len__(self):
        'Denotes the total number of samples'
        return (len(self.fg_files))
        
    def __getitem__(self, index):
        'Generates one sample of data'      
        # print(index)
        if random.random() < 0.5:
            image = cv2.imread(self.img_path + str(index) + '.jpg')
            mask  = cv2.imread(self.msk_path + str(index) + '.png')
        else:
            rand_index = random.randint(1, 5)
            image, mask = self.cropped(rand_index)
        
        if random.random() < 0.5:
            image = cv2.GaussianBlur(image, (5,5), 0)
            
        
        if random.random() < 0.5:
            image = cv2.flip(image, 0)
            mask = cv2.flip(mask, 0)            
        
        label = self.label_mask(mask)
        
        
        # introduce normalization
        if self.transform:
            inp_data = self.transform(image)
        
        inp_label = torch.from_numpy(label)
    
        return inp_data.float(), inp_label.float()

# test dataloader with my images. 

class Dataset_unet_valid(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, list_IDs, transform = None):
        'Initialization'
        self.list_IDs = list_IDs
        self.transform = transform
        
        self.img_path = 'images/datasets_2/valid/img/'
        self.msk_path = 'images/datasets_2/valid/msk/'

        self.fg_files = os.listdir(self.img_path)

    def label_mask(self, mask):
        newMask_1 = np.zeros(mask.shape[:2])
        newMask_1[mask[:,:,0] == 163] = 1

        newMask_2 = np.zeros(mask.shape[:2])
        newMask_2[mask[:,:,0] == 28] = 1

        newMask_3 = np.zeros(mask.shape[:2])
        newMask_3[mask[:,:,0] == 110] = 1

        newMask_4 = np.zeros(mask.shape[:2])
        newMask_4[mask[:,:,0] == 39] = 1

        mask = np.asarray([newMask_1, newMask_2, newMask_3, newMask_4]).astype(np.float32)

        return mask
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.fg_files) 
        
    def __getitem__(self, index):
        'Generates one sample of data'      
        image = cv2.imread(self.img_path + str(index) + '.jpg')
        mask  = cv2.imread(self.msk_path + str(index) + '.png')
        
        label = self.label_mask(mask)
        
        if self.transform:
            inp_data = self.transform(image)
            
        inp_label = torch.from_numpy(label)        
        
        return inp_data.float(), inp_label.float()

