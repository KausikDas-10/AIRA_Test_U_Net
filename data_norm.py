# Mean and STD calculate
import cv2 
import torch
import numpy as np

img_1 = cv2.imread('AIRA_Images/Images/1.jpg')
img_2 = cv2.imread('AIRA_Images/Images/2.jpg')
img_3 = cv2.imread('AIRA_Images/Images/3.jpg')
img_4 = cv2.imread('AIRA_Images/Images/4.jpg')
img_5 = cv2.imread('AIRA_Images/Images/5.jpg')
img_6 = cv2.imread('AIRA_Images/Images/6.jpg')
print(img_1.shape)
img = np.asarray([img_1, img_2, img_3, img_4, img_5, img_6]).astype(np.float32)

img_tr = torch.from_numpy(img)
img_tr = img_tr.float().div(255.0)

print('R - mean :', img_tr[:,:,:,0].mean())
print('R - std  :', img_tr[:,:,:,0].std())

print('G - mean :', img_tr[:,:,:,1].mean())
print('G - std  :', img_tr[:,:,:,1].std())

print('B - mean :', img_tr[:,:,:,2].mean())
print('B - std  :', img_tr[:,:,:,2].std())
