'''
1. Crop image patches of size 224 x 224 from the big size image
2. From each image I extract 5000 patches. 

2. Cropping are done two ways:
    i. non-overlapping sliding window based
    ii. randomly find regions and cropped

3. Data-augmentations are:
    i. Randomly crop patches of 224 x 224 
    ii. Image rotation (90, 180 and 270)
    iii. Image flipping (vertical and horizonal)

''' 

import cv2
import numpy as np
import os 

step_count = 0

def crop_img_mask_train(image, mask):
    crop_height = 224
    crop_width  = 224
    
    step_count = len(os.listdir('images/datasets_2/train/img/'))
    count = step_count
    
    for i in range(0, image.shape[0]-224, 224):
        for j in range(0, image.shape[1]-224, 224):
    
            crop_image = image[i:i + crop_height, j:j + crop_width]
            crop_mask= mask[i:i + crop_height, j:j + crop_width]

            filename_img = 'images/datasets_2/train/img/' + str(count) + '.jpg'
            filename_msk = 'images/datasets_2/train/msk/' + str(count) + '.png'
            
            cv2.imwrite(filename_img, crop_image)
            cv2.imwrite(filename_msk, crop_mask)
            
            count = count + 1

    while(True):    

        max_x = image.shape[1] - crop_width
        max_y = image.shape[0] - crop_height

        x = np.random.randint(0, max_x)
        y = np.random.randint(0, max_y)

        crop_image = image[y: y + crop_height, x: x + crop_width]
        crop_mask  = mask[y: y + crop_height, x: x + crop_width]

        filename_img = 'images/datasets_2/train/img/' + str(count) + '.jpg'
        filename_msk = 'images/datasets_2/train/msk/' + str(count) + '.png'
            
        cv2.imwrite(filename_img, crop_image)
        cv2.imwrite(filename_msk, crop_mask)
        
        count = count + 1

        if count > step_count + 250:
            break
            

    while(True):    

        max_x = image.shape[1] - crop_width
        max_y = image.shape[0] - crop_height

        x = np.random.randint(0, max_x)
        y = np.random.randint(0, max_y)

        crop_image = image[y: y + crop_height, x: x + crop_width]
        crop_mask  = mask[y: y + crop_height, x: x + crop_width]

        # Start augmentation
        rotate_image = cv2.rotate(crop_image, cv2.ROTATE_90_CLOCKWISE)
        rotate_mask  = cv2.rotate(crop_mask, cv2.ROTATE_90_CLOCKWISE)
        
        filename_img = 'images/datasets_2/train/img/' + str(count) + '.jpg'
        filename_msk = 'images/datasets_2/train/msk/' + str(count) + '.png'
            
        cv2.imwrite(filename_img, rotate_image)
        cv2.imwrite(filename_msk, rotate_mask)
        
        count = count + 1

        rotate_image = cv2.rotate(crop_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        rotate_mask  = cv2.rotate(crop_mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        filename_img = 'images/datasets_2/train/img/' + str(count) + '.jpg'
        filename_msk = 'images/datasets_2/train/msk/' + str(count) + '.png'
            
        cv2.imwrite(filename_img, rotate_image)
        cv2.imwrite(filename_msk, rotate_mask)
        
        count = count + 1

        rotate_image = cv2.rotate(crop_image, cv2.ROTATE_180)
        rotate_mask  = cv2.rotate(crop_mask, cv2.ROTATE_180)
        
        filename_img = 'images/datasets_2/train/img/' + str(count) + '.jpg'
        filename_msk = 'images/datasets_2/train/msk/' + str(count) + '.png'
            
        cv2.imwrite(filename_img, rotate_image)
        cv2.imwrite(filename_msk, rotate_mask)

        count = count + 1
        
        rotate_image = cv2.flip(crop_image, 0)
        rotate_mask  = cv2.flip(crop_mask, 0)
        
        filename_img = 'images/datasets_2/train/img/' + str(count) + '.jpg'
        filename_msk = 'images/datasets_2/train/msk/' + str(count) + '.png'
            
        cv2.imwrite(filename_img, rotate_image)
        cv2.imwrite(filename_msk, rotate_mask)

        count = count + 1

        rotate_image = cv2.flip(crop_image, 1)
        rotate_mask  = cv2.flip(crop_mask, 1)
        
        filename_img = 'images/datasets_2/train/img/' + str(count) + '.jpg'
        filename_msk = 'images/datasets_2/train/msk/' + str(count) + '.png'
            
        cv2.imwrite(filename_img, rotate_image)
        cv2.imwrite(filename_msk, rotate_mask)

        count = count + 1

        rotate_image = cv2.flip(crop_image, -1)
        rotate_mask  = cv2.flip(crop_mask, -1)
        
        filename_img = 'images/datasets_2/train/img/' + str(count) + '.jpg'
        filename_msk = 'images/datasets_2/train/msk/' + str(count) + '.png'
            
        cv2.imwrite(filename_img, rotate_image)
        cv2.imwrite(filename_msk, rotate_mask)
        
        count = count + 1
        
        if count > step_count + 5000:
            break

    
    print(count)

def crop_img_mask_valid(image, mask):
    crop_height = 224
    crop_width  = 224
    count = 0
    
    for i in range(0, image.shape[0]-224, 224):
        for j in range(0, image.shape[1]-224, 224):
    
            crop_image = image[i:i + crop_height, j:j + crop_width]
            crop_mask= mask[i:i + crop_height, j:j + crop_width]

            filename_img = 'images/datasets_2/valid/img/' + str(count) + '.jpg'
            filename_msk = 'images/datasets_2/valid/msk/' + str(count) + '.png'
            
            cv2.imwrite(filename_img, crop_image)
            cv2.imwrite(filename_msk, crop_mask)
            
            count = count + 1


# Train images prep. only
for i in range(1, 6):
    img = cv2.imread('AIRA_Images/Images/' + str(i) + '.jpg')
    mask = cv2.imread('AIRA_Images/Images/' + str(i)+ '_label.png')

    crop_img_mask_train(img, mask)

# Valid images prep. 
valid_img_idx = 6
img = cv2.imread('AIRA_Images/Images/' + str(valid_img_idx) + '.jpg')
mask = cv2.imread('AIRA_Images/Images/' + str(valid_img_idx)+ '_label.png')

crop_img_mask_valid(img, mask)