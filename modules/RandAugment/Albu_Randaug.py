# code in this file is adpated from rpmcruz/autoaugment
# https://github.com/rpmcruz/autoaugment/blob/master/transformations.py
import random
import numpy as np
import torch

import albumentations as albu
from albumentations.augmentations.geometric.transforms import Affine 

################################################## Paper 
def Identity(img):
    return img

def ShearX(img):  # [-40, 40]
    img = Affine(shear={"x": (-40, 40), "y": (0, 0)}, interpolation=0, p=1.0)(image=img)['image']
    return img  

def ShearY(img):  # [-40, 40]
    img = Affine(shear={"x": (-0, 0), "y": (-40, 40)}, interpolation=0, p=1.0)(image=img)['image']
    return img          

def TranslateX(img):  # [-150, 150] => percentage: [-0.45, 0.45]
    img = Affine(translate_percent={"x": (-0.45, 0.45), "y": (-0, 0)}, interpolation=0, p=1.0)(image=img)['image']
    return img 

def TranslateY(img):  # [-150, 150] => percentage: [-0.45, 0.45]
    img = Affine(translate_percent={"x": (-0, 0), "y": (-0.45, 0.45)}, interpolation=0, p=1.0)(image=img)['image']
    return img 

def Equalize(img):
    img = albu.Equalize(p=1.0)(image=img)['image']
    return img 

def Solarize(img):  # [0, 256]
    img = albu.Solarize(threshold=[0, 256], p=1.0)(image=img)['image']
    return img 

def Posterize(img):  # [4, 8]
    img = albu.Posterize(num_bits=[4, 8], p=1.0)(image=img)['image']
    return img     



################################################## from my insight!

def ColorJitter(img):  # [0.1,1.9]
    img = albu.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4, p=1.0)(image=img)['image']
    return img  


def ShiftScaleRotate(img):  # [-30, 30]
    img = albu.ShiftScaleRotate(scale_limit=(-0.4, 0.4), rotate_limit=(-45, 45), shift_limit=0.0, p=1, border_mode=0)(image=img)['image']
    return img 


def BrightnessContrast (img):  # [0.1,1.9]
    img = albu.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), brightness_by_max=True, p=1.0)(image=img)['image']
    return img       


def Sharpen(img):  # [0.1,1.9]
    img = albu.Sharpen(alpha=(0.2, 0.4), lightness=(0.2, 0.4), p=1.0)(image=img)['image']
    return img        


def HorizontalFlip(img):  # not from the paper
    img = albu.HorizontalFlip(p=1.0)(image=img)['image']
    return img 


def GaussianNoise(img):  
    img = albu.GaussNoise(var_limit=(60.0, 100.0), mean=0, always_apply=False, p=1.0)(image=img)['image']
    return img


def Blur(img):  
    img = albu.OneOf([albu.Blur(blur_limit=(11, 13), p=1), albu.MotionBlur(blur_limit=(11, 13), p=1), albu.MedianBlur(blur_limit=(11, 13), p=0.1)], p=1)(image=img)['image']
    return img    

def CLAHE(img):
    img = albu.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), always_apply=False, p=1.0)(image=img)['image']
    return img 

def Invert(img):
    img = albu.InvertImg(p=1.0)(image=img)['image']
    return img 

def CoarseDropout(img):  
    img  = albu.CoarseDropout(max_holes=14, max_height=img.shape[0]//4, max_width=img.shape[0]//4, min_holes=12, min_height=img.shape[0]//5, min_width=img.shape[0]//5, fill_value=0, mask_fill_value=None, always_apply=False, p=1.0)(image=img)['image']
    return img



#### 뭔가 갯수랑 강도를 올리고 싶음. 
def augment_list():  # 16 oeprations and their ranges

    # https://github.com/tensorflow/tpu/blob/8462d083dd89489a79e3200bcc8d4063bf362186/models/official/efficientnet/autoaugment.py#L505
    l = [
        # from Paper
        Identity,        
        # Equalize,
        # Posterize,
        # Solarize,
        ShearX,
        ShearY,
        TranslateX,
        TranslateY,
         
        # from my insight
        # BrightnessContrast,
        # Invert,
        Sharpen,
        ShiftScaleRotate,
        GaussianNoise,
        # Blur,
        # ColorJitter,
        # CLAHE,
        # CoarseDropout,
        ]

    return l


class Albu_RandAugment:
    def __init__(self, n):
        self.n = n
        self.augment_list = augment_list()

    def __call__(self, img, **kwargs):
        ops = random.sample(self.augment_list, k=self.n)
        
        # Must add
        ops += [Blur]
        ops += [CoarseDropout]
        
        for op in ops:
            # print("Used... ", op.__name__)
            img = op(img)

        return img
