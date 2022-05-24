import torch
import numpy as np
import pandas as pd
import functools
import cv2 
import re
import glob

import skimage
from PIL import Image

import albumentations as albu
from monai.transforms import *
from monai.data import Dataset


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


def get_path(mode, path, x):
    return path + '/' + mode + '/' + str(x) + '.png'


def get_label(x):
    return np.float32(x)


def get_png(path):
    image = np.array(Image.open(path)).astype('float32')

    if (len(image.shape) !=2):
        image = image[..., 0]
    
    assert len(image.shape) == 2, print('에러 파일=', path)
    
    return image
    

def resize_and_padding_with_aspect_clahe(image, spatial_size):
    image = np.clip(image, a_min=np.percentile(image, 0.5), a_max=np.percentile(image, 99.5))
    
    image -= image.min()
    image /= image.max()                   # clahe 전에 필요
    image = skimage.img_as_ubyte(image)

    image = albu.PadIfNeeded(min_height=max(image.shape), min_width=max(image.shape), always_apply=True, border_mode=0)(image=image)['image']
    image = cv2.resize(image, spatial_size, interpolation=cv2.INTER_CUBIC)
    image = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(image)
    
    image = skimage.util.img_as_float32(image)  # clahe 후에 필요
    return image


def Albu_2D_Transform_Compose(input):
    image = input['image'].squeeze(0)

    Trans = albu.Compose([
                    albu.OneOf([
                        albu.MedianBlur(blur_limit=3, p=0.1),
                        albu.MotionBlur(p=0.2),
                        ], p=0.2),
                    albu.OneOf([
                        albu.OpticalDistortion(p=0.3),
                        ], p=0.2),
                    albu.OneOf([
                        albu.GaussNoise(p=0.2),
                        albu.MultiplicativeNoise(p=0.2),
                        ], p=0.2),
                    albu.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=0, val_shift_limit=0.1, p=0.3),
                    ])
    augment = Trans(image=image)
    image = augment['image']
    input['image'] = np.expand_dims(image, axis=0)

    return input


def minmax_normalize(image, option=False):
    if len(np.unique(image)) != 1:  # Sometimes it cause the nan inputs...
        image -= image.min()
        image /= image.max() 

    if option:
        image = (image - 0.5) / 0.5  # Range -1.0 ~ 1.0   @ We do not use -1~1 range becuase there is no Tanh act.

    return image.astype('float32')


def default_collate_fn(batch):
    batch = list(filter(lambda x: torch.isnan(x['image'].max()).item() == False, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def RSNA_BAA_Dataset(mode, data_folder_dir="/mnt/nas125_vol2/kanggilpark/child/bone_age/data"):  
    train_transforms = Compose(
        [
            # Just png Load
            Lambdad(keys=["image"], func=get_png),
            Lambdad(keys=["image"], func=functools.partial(resize_and_padding_with_aspect_clahe, spatial_size=(256, 256))),                                       
            AddChanneld(keys=["image", "label", "gender"]),              

            # (45 degree rotation, vertical & horizontal flip & scaling)
            RandFlipd(keys=["image"], prob=0.1, spatial_axis=[0, 1], allow_missing_keys=False),
            RandRotate90d(keys=["image"], prob=0.1, spatial_axes=[0, 1], allow_missing_keys=False), # 추가
            RandRotated(keys=["image"], prob=0.1, range_x=np.pi/12, range_y=np.pi/12, range_z=0.0, keep_size=True, align_corners=False, allow_missing_keys=False),
            RandZoomd(keys=["image"], prob=0.1, min_zoom=0.9, max_zoom=1.1, align_corners=None, keep_size=True, allow_missing_keys=False), # min : 0.5 -> 0.9, max : 2 -> 1.1 수정
            
            # Additional Augmentation
            Albu_2D_Transform_Compose, 

            # Normalize
            Lambdad(keys=["image"], func=functools.partial(minmax_normalize, option=False)),                  
            ToTensord(keys=["image"]),
        ]
    )
    valid_transforms = Compose(
        [
            # Just png Load
            Lambdad(keys=["image"], func=get_png),
            Lambdad(keys=["image"], func=functools.partial(resize_and_padding_with_aspect_clahe, spatial_size=(256, 256))),                                       
            AddChanneld(keys=["image", "label", "gender"]),          

            # Normalize
            Lambdad(keys=["image"], func=functools.partial(minmax_normalize, option=False)),                  
            ToTensord(keys=["image"]),
        ]
    )   
    
    if mode == 'train':
        train_df = pd.read_csv(data_folder_dir + "/train.csv")
        img_list    = list(map(functools.partial(get_path, mode, data_folder_dir),train_df['id']))
        age_list    = list(map(get_label, train_df['boneage']))
        gender_list = list(map(get_label, train_df['male']))
        print("Train [Total]  number = ", len(img_list))
        transform_combination = train_transforms

    elif mode == 'valid':
        valid_df = pd.read_csv(data_folder_dir + "/valid.csv")
        img_list    = list(map(functools.partial(get_path, mode, data_folder_dir),valid_df['id']))
        age_list    = list(map(get_label, valid_df['boneage']))
        gender_list = list(map(get_label, valid_df['male']))
        print("Valid [Total]  number = ", len(img_list))
        transform_combination = valid_transforms

    data_dicts   = [ {"image": image_name, "label": study_label_name, 'gender': gender} for image_name, study_label_name, gender in zip(img_list, age_list, gender_list)]
    return Dataset(data=data_dicts, transform=transform_combination), default_collate_fn


# TEST
def RSNA_BAA_Dataset_TEST(data_folder_dir="/mnt/nas125_vol2/kanggilpark/child/bone_age/data"):
    transforms = Compose(
        [
            # Just png Load
            Lambdad(keys=["image"], func=get_png),
            Lambdad(keys=["image"], func=functools.partial(resize_and_padding_with_aspect_clahe, spatial_size=(256, 256))),                                       
            AddChanneld(keys=["image" "label", "gender"]),           

            # Normalize
            Lambdad(keys=["image"], func=functools.partial(minmax_normalize, option=False)),                  
            ToTensord(keys=["image"]),
        ]
    )
    test_df  = pd.read_csv(data_folder_dir + "/test.csv")
    img_list    = list(map(functools.partial(get_path, 'test', data_folder_dir),test_df['id']))
    age_list    = list(map(get_label, test_df['boneage']))
    gender_list = list(map(get_label, test_df['male']))
    print("test [Total]  number = ", len(img_list))
    transform_combination = transforms

    data_dicts   = [ {"image": image_name, "label": study_label_name, 'gender': gender} for image_name, study_label_name, gender in zip(img_list, age_list, gender_list)]
    return Dataset(data=data_dicts, transform=transform_combination), default_collate_fn
