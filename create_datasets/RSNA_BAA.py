import torch
import numpy as np
import pandas as pd
import functools
import cv2 
import re
import glob

import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut
import SimpleITK as sitk

import albumentations as albu
from monai.transforms import *
from monai.data import Dataset

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

def get_pixels_hu(path):
    # pydicom version...!
    # referred from https://www.kaggle.com/gzuidhof/full-preprocessing-tutorial
    # ref: pydicom.pixel_data_handlers.util.apply_modality_lut
    # '''
    # Awesome pydicom lut fuction...!
    # ds  = pydicom.dcmread(fname)
    # arr = ds.pixel_array
    # hu  = apply_modality_lut(arr, ds)
    # '''
    dcm_image = pydicom.dcmread(path)
    try:
        image  = dcm_image.pixel_array    
    except:
        print("Error == ", path)    
        
    try:
        image  = apply_modality_lut(image, dcm_image)        
    except:
        image = image.astype(np.int16)
        image[image == -2000] = 0

        intercept = dcm_image.RescaleIntercept
        slope     = dcm_image.RescaleSlope

        if slope != 1:
            image = slope * image.astype(np.float64)
            image = image.astype(np.int16)

        image += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)

def dicom_normalize(image): 
    if len(np.unique(image)) != 1:  # Sometimes it cause the nan inputs...
        image -= image.min()
        image /= image.max() 

    return image.astype('float32')

def dicom_resize_and_padding_with_aspect(image, spatial_size):
    image = np.clip(image, a_min=np.percentile(image, 0.5), a_max=np.percentile(image, 99.5))
    image = albu.PadIfNeeded(min_height=max(image.shape), min_width=max(image.shape), always_apply=True, border_mode=0)(image=image)['image']
    image = cv2.resize(image, spatial_size, interpolation=cv2.INTER_CUBIC)
    return image

def minmax_normalize(image, option=False):
    if len(np.unique(image)) != 1:  # Sometimes it cause the nan inputs...
        image -= image.min()
        image /= image.max() 

    if option:
        image = (image - 0.5) / 0.5  # Range -1.0 ~ 1.0   @ We do not use -1~1 range becuase there is no Tanh act.

    return image.astype('float32')



# def default_collate_fn(batch):
#     batch = list(filter(lambda x: x is not None, batch))
#     return torch.utils.data.dataloader.default_collate(batch)

def default_collate_fn(batch):
    batch = list(filter(lambda x: torch.isnan(x['image'].max()).item() == False, batch))
    return torch.utils.data.dataloader.default_collate(batch)








def RSNA_BAA_Dataset(mode, data_folder_dir="/workspace/sunggu/1.Hemorrhage/SMART-Net/datasets/samples"):  
    if mode == 'train':
        img_list     = list_sort_nicely(glob.glob(data_folder_dir + "/train/*_img.nii.gz"))
        label_list   = list_sort_nicely(glob.glob(data_folder_dir + "/train/*_mask.nii.gz"))
        data_dicts   = [{"image": image_name, "image_path": image_name, "label": label_name, "label_path": label_name} for image_name, label_name in zip(img_list, label_list)]        

        print("Train [Total]  number = ", len(img_list))
        print("Train [Hemo]   number = ", len([i for i in img_list if "_hemo_" in i]))
        print("Train [Normal] number = ", len([i for i in img_list if "_normal_" in i]))

        transforms = Compose(
            [
                # Just dicom Load
                Lambdad(keys=["image"], func=get_pixels_hu),
                Lambdad(keys=["image"], func=functools.partial(dicom_resize_and_padding_with_aspect, spatial_size=(256, 256))),                    
                Lambdad(keys=["image"], func=dicom_normalize),                    
                AddChanneld(keys=["image"]),              

                # (45 degree rotation, vertical & horizontal flip & scaling)
                RandFlipd(keys=["image"], prob=0.1, spatial_axis=[0, 1], allow_missing_keys=False),
                RandRotated(keys=["image"], prob=0.1, range_x=np.pi/4, range_y=np.pi/4, range_z=0.0, keep_size=True, align_corners=False, allow_missing_keys=False),
                RandZoomd(keys=["image"], prob=0.1, min_zoom=0.5, max_zoom=2.0, align_corners=None, keep_size=True, allow_missing_keys=False),
                
                # Normalize
                Lambdad(keys=["image"], func=functools.partial(minmax_normalize, option=False)),                  
                ToTensord(keys=["image"]),
            ]
        )     


    else :
        img_list     = list_sort_nicely(glob.glob(data_folder_dir + "/valid/*_img.nii.gz"))
        label_list   = list_sort_nicely(glob.glob(data_folder_dir + "/valid/*_mask.nii.gz"))
        data_dicts   = [{"image": image_name, "image_path": image_name, "label": label_name, "label_path": label_name} for image_name, label_name in zip(img_list, label_list)]        

        print("Valid [Total]  number = ", len(img_list))
        print("Valid [Hemo]   number = ", len([i for i in img_list if "_hemo_" in i]))
        print("Valid [Normal] number = ", len([i for i in img_list if "_normal_" in i]))

        transforms = Compose(
            [
                # Just dicom Load
                Lambdad(keys=["image"], func=get_pixels_hu),
                Lambdad(keys=["image"], func=functools.partial(dicom_resize_and_padding_with_aspect, spatial_size=(256, 256))),                    
                Lambdad(keys=["image"], func=dicom_normalize),                    
                AddChanneld(keys=["image"]),          

                # Data Normalize
                # Lambdad(keys=["image"], func=minmax_normalize),    # it is the same as the 'docom_normalize'                
                ToTensord(keys=["image"]),
            ]
        )         
        
    return Dataset(data=data_dicts, transform=transforms), default_collate_fn




# TEST
def RSNA_BAA_Dataset_TEST(data_folder_dir="/workspace/sunggu/1.Hemorrhage/SMART-Net/datasets/samples"):
    img_list     = list_sort_nicely(glob.glob(data_folder_dir + "/test/*_img.nii.gz"))
    label_list   = list_sort_nicely(glob.glob(data_folder_dir + "/test/*_mask.nii.gz"))
    data_dicts   = [{"image": image_name, "image_path": image_name, "label": label_name, "label_path": label_name} for image_name, label_name in zip(img_list, label_list)]        

    print("TEST [Total]  number = ", len(img_list))
    print("TEST [Hemo]   number = ", len([i for i in img_list if "_hemo_" in i]))
    print("TEST [Normal] number = ", len([i for i in img_list if "_normal_" in i]))

    transforms = Compose(
        [
            # Just dicom Load
            Lambdad(keys=["image"], func=get_pixels_hu),
            Lambdad(keys=["image"], func=functools.partial(dicom_resize_and_padding_with_aspect, spatial_size=(256, 256))),                    
            Lambdad(keys=["image"], func=dicom_normalize),                    
            AddChanneld(keys=["image"]),          

            # Data Normalize
            # Lambdad(keys=["image"], func=minmax_normalize),    # it is the same as the 'docom_normalize'                
            ToTensord(keys=["image"]),
        ]
    )         
        
    return Dataset(data=data_dicts, transform=transforms), default_collate_fn
