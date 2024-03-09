import torch
import numpy as np
import pandas as pd
import functools
import cv2 
import re
import glob
import pydicom
import skimage
from PIL import Image
import albumentations as A
from torch.utils.data import Dataset as BaseDataset
from albumentations.pytorch import ToTensorV2

# Pydicom Error shut down
import warnings
warnings.filterwarnings(action='ignore') 


# functions
def list_sort_nicely(l):   
    def tryint(s):        
        try:            
            return int(s)        
        except:            
            return s
        
    def alphanum_key(s):
        return [ tryint(c) for c in re.split('([0-9]+)', s) ]
    l.sort(key=alphanum_key)    
    return l

def change_to_uint8(image, **kwargs):
    return skimage.util.img_as_ubyte(image)

def change_to_float32(image, **kwargs):
    return skimage.util.img_as_float32(image)

def Clip_Resize_PaddingWithAspect(image, **kwargs):
    image = np.clip(image, a_min=np.percentile(image, 0.5), a_max=np.percentile(image, 99.5))
    image = A.PadIfNeeded(min_height=max(image.shape), min_width=max(image.shape), always_apply=True, border_mode=0)(image=image)['image']
    image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_CUBIC)
    return image

def minmax_normalize(image, option=False, **kwargs):
    if len(np.unique(image)) != 1:  # Sometimes it cause the nan inputs...
        image -= image.min()
        image /= image.max() 

    if option:
        image = (image - 0.5) / 0.5  # Range -1.0 ~ 1.0   @ We do not use -1~1 range becuase there is no Tanh act.

    return image.astype('float32')

def fixed_clahe(image, **kwargs):
    clahe_mat = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    if len(image.shape) == 2 or image.shape[2] == 1:
        image = clahe_mat.apply(image)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        image[:, :, 0] = clahe_mat.apply(image[:, :, 0])
        image = cv2.cvtColor(image, cv2.COLOR_LAB2RGB)
    return image
    

def get_transforms(mode="train"):
    # medical augmentation
    if mode == "train":
        return A.Compose([
            # preprocessing
            A.Lambda(image=Clip_Resize_PaddingWithAspect),

            # clahe
            A.Lambda(image=minmax_normalize, always_apply=True),
            A.Lambda(image=change_to_uint8, always_apply=True),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), always_apply=True),
            A.Lambda(image=change_to_float32, always_apply=True),
            A.Lambda(image=minmax_normalize, always_apply=True),

            # augmentation
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(scale_limit=0.15, rotate_limit=15, shift_limit=0.1, border_mode=0, p=1.0),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.05, contrast_limit=0.05, p=1.0),
                A.RandomBrightness(limit=0.05, p=1.0),
                A.RandomContrast(limit=0.05, p=1.0),
                ], p=0.5),
            A.OneOf([
                A.GaussianBlur(blur_limit=3, sigma_limit=0, p=1.0),
                A.MedianBlur(blur_limit=3, p=1.0),
                A.MotionBlur(blur_limit=3, p=1.0),
                ], p=0.5),
            A.GaussNoise(var_limit=(0.00001, 0.00005), mean=0, per_channel=True, p=0.5),
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.8),    # https://arxiv.org/pdf/2212.04690.pdf
            
            # normalization
            ToTensorV2(transpose_mask=True)
        ], additional_targets={'image2':'image'})
    
    elif mode == "valid":
        return A.Compose([
            # preprocessing
            A.Lambda(image=Clip_Resize_PaddingWithAspect),
            
            # clahe
            A.Lambda(image=minmax_normalize, always_apply=True),
            A.Lambda(image=change_to_uint8, always_apply=True),
            A.Lambda(image=fixed_clahe, always_apply=True),
            A.Lambda(image=change_to_float32, always_apply=True),
            A.Lambda(image=minmax_normalize, always_apply=True),

            # normalization
            ToTensorV2(transpose_mask=True)
        ], additional_targets={'image2':'image'})

    elif mode == "test":
        return A.Compose([
            # preprocessing
            A.Lambda(image=Clip_Resize_PaddingWithAspect),
            
            # clahe
            A.Lambda(image=minmax_normalize, always_apply=True),
            A.Lambda(image=change_to_uint8, always_apply=True),
            A.Lambda(image=fixed_clahe, always_apply=True),
            A.Lambda(image=change_to_float32, always_apply=True),
            A.Lambda(image=minmax_normalize, always_apply=True),

            # normalization
            ToTensorV2(transpose_mask=True)
        ], additional_targets={'image2':'image'})        


class GRAZPEDWRI_DX_Dataset(BaseDataset):
    def __init__(self, mode="train"):
        self.mode = mode
        if mode == 'train':
            target_df = pd.read_csv("/workspace/sunggu/3.Child/PedXnet_Code_Factory/datasets/4.GRAZPEDWRI-DX/train.csv")
            self.image_list = target_df['filestem'].values
            self.label_list = target_df['fracture_visible'].values
        elif mode == 'valid':
            target_df = pd.read_csv("/workspace/sunggu/3.Child/PedXnet_Code_Factory/datasets/4.GRAZPEDWRI-DX/valid.csv")
            self.image_list = target_df['filestem'].values
            self.label_list = target_df['fracture_visible'].values
        elif mode == 'test':
            target_df = pd.read_csv("/workspace/sunggu/3.Child/PedXnet_Code_Factory/datasets/4.GRAZPEDWRI-DX/test.csv")
            self.image_list = target_df['filestem'].values
            self.label_list = target_df['fracture_visible'].values            
        
        self.transforms = get_transforms(mode=mode)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, i):

        # Read Image
        image = np.array(Image.open(self.image_list[i])) # uint8, 0~255 range
        label = torch.tensor(self.label_list[i]).unsqueeze(0)
        
        # print("Check!", image.shape, image.dtype, image.max(), image.min())

        # Apply transforms
        sample = self.transforms(image=image)
        image  = sample['image']
        
        # image [B, 1, 512, 512], label [B, 1]
        if self.mode == 'test':
            return image.float(), label.float(), self.image_list[i]
        else:
            return image.float(), label.float()        