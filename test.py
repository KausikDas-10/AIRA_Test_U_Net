import torch
import torch.nn as nn
from unet_architecture.unet import UNet

import matplotlib.pyplot as plt
from torchvision import transforms, datasets, models

import torch.nn.functional as F

import numpy as np
import cv2

import argparse
parser = argparse.ArgumentParser(description = 'Display the prediction on test images')
parser.add_argument('-img_idx', '--img_idx', type=int, help='int index between 0 to 23')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_class = 4

# load trained model
model = UNet(num_class).to(device)

pretrain = 'trained_model/unet_vanila_best.pth'
model.load_state_dict(torch.load(pretrain))
model = model.eval()
model = model.cuda()

trans_valid = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.6234, 0.4796, 0.7374], [0.1990, 0.2924, 0.1550])
])

img = cv2.imread('images/datasets_2/valid/img/' + str(args.img_idx) + '.jpg')
mask  = cv2.imread('images/datasets_2/valid/msk/' + str(args.img_idx) + '.png')
img = torch.unsqueeze(trans_valid(img), 0)
img = img.float()
# plt.imshow(mask)

# Predict 
op = F.sigmoid(model(img.to(device)))
op_np = op.cpu().detach().numpy()

# Predicton lable in terms of each class 
pred_mask_1 = np.zeros([224, 224])
pred_mask_2 = np.zeros([224, 224])
pred_mask_3 = np.zeros([224, 224])
pred_mask_4 = np.zeros([224, 224])

th = 0.5

pred_mask_1[op_np[0, 0, :, :] > th] = 1
pred_mask_2[op_np[0, 1, :, :] > th] = 1
pred_mask_3[op_np[0, 2, :, :] > th] = 1
pred_mask_4[op_np[0, 3, :, :] > th] = 1

# test generate image process ... 
# Funtion for labels mask preparation. 
def label_mask(mask):
    
    newMask_1 = np.zeros(mask.shape[:2])
    newMask_1[mask[:, :, 0] == 163] = 1

    newMask_2 = np.zeros(mask.shape[:2])
    newMask_2[mask[:, :, 0] == 28] = 1

    newMask_3 = np.zeros(mask.shape[:2])
    newMask_3[mask[:, :, 0] == 110] = 1

    newMask_4 = np.zeros(mask.shape[:2])
    newMask_4[mask[:, :, 0] == 39] = 1
    
    mask = np.asarray([newMask_1, newMask_2, newMask_3, newMask_4]).astype(np.float32)
    
    return mask, newMask_1, newMask_2, newMask_3, newMask_4

_, gt_mask_1, gt_mask_2, gt_mask_3, gt_mask_4 = label_mask(mask)

f, axarr = plt.subplots(4,2, figsize = (20, 20))

axarr[0,0].imshow((gt_mask_1 * 255).astype(np.uint8))
axarr[0,0].set_title('GT CLASS-1')
axarr[0,1].imshow((pred_mask_1 * 255).astype(np.uint8))
axarr[0,1].set_title('PRED CLASS-1')

axarr[1,0].imshow(gt_mask_2)
axarr[1,0].set_title('GT CLASS-2')
axarr[1,1].imshow(pred_mask_2)
axarr[1,1].set_title('PRED CLASS-2')

axarr[2,0].imshow(gt_mask_3)
axarr[2,0].set_title('GT CLASS-3')
axarr[2,1].imshow(pred_mask_3)
axarr[2,1].set_title('PRED CLASS-3')

axarr[3,0].imshow(gt_mask_4)
axarr[3,0].set_title('GT CLASS-4')
axarr[3,1].imshow(pred_mask_4)
axarr[3,1].set_title('PRED CLASS-4')
plt.show()