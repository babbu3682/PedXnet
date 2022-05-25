import torch
import numpy as np
import pandas as pd
import skimage

import SimpleITK as sitk
import pydicom

from monai.transforms import *
from monai.data import Dataset
import albumentations as albu
import cv2 
import functools

import warnings
warnings.filterwarnings(action='ignore') 

from pydicom.pixel_data_handlers.util import apply_modality_lut

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

def add_img_path(x):
    path = x.replace('/workspace/data/', '/home/pkg777774/child_xray/Cleansing_X-Ray/')  # 경로 변경 
    return path

def label_class(x):
    cls_xray = {'Chest':0, 'Head':1, 'Abdomen':2, 'Hand':3, 'Foot':4, 'Knee':5, 'Neck':6, 'Spine':7, 'Elbow':8, 'Lower leg':9, 'Pelvis':10,
                'Thigh':11, 'Hip':12, 'Forearm':13, 'Shoulder':14, 'Upper arm':15}
    label = cls_xray[x]
    return label

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

def Supervised_16Class(mode, data_folder_dir='/home/pkg777774/child_xray/fracture/jykim'):
    train_transforms = Compose(
        [
            # Just dicom Load
            Lambdad(keys=["image"], func=get_pixels_hu),
            Lambdad(keys=["image"], func=functools.partial(dicom_resize_and_padding_with_aspect, spatial_size=(256, 256))),                    
            Lambdad(keys=["image"], func=dicom_normalize),                    
            AddChanneld(keys=["image"]),              

            # (45 degree rotation, vertical & horizontal flip & scaling)
            RandFlipd(keys=["image"], prob=0.1, spatial_axis=[0, 1], allow_missing_keys=False),
            RandRotate90d(keys=["image"], prob=0.1, spatial_axes=[0, 1], allow_missing_keys=False), # 추가
            RandRotated(keys=["image"], prob=0.1, range_x=np.pi/12, range_y=np.pi/12, range_z=0.0, keep_size=True, align_corners=False, allow_missing_keys=False),
            RandZoomd(keys=["image"], prob=0.1, min_zoom=0.9, max_zoom=1.1, align_corners=None, keep_size=True, allow_missing_keys=False), # min : 0.5 -> 0.9, max : 2 -> 1.1 수정

            # Normalize
            Lambdad(keys=["image"], func=functools.partial(minmax_normalize, option=False)),                  
            ToTensord(keys=["image"]),
        ]
    )
    valid_transforms = Compose(
        [
            # Just dicom Load
            Lambdad(keys=["image"], func=get_pixels_hu),
            Lambdad(keys=["image"], func=functools.partial(dicom_resize_and_padding_with_aspect, spatial_size=(256, 256))),                    
            Lambdad(keys=["image"], func=dicom_normalize),                    
            AddChanneld(keys=["image"]),          

            # Data Normalize               
            ToTensord(keys=["image"]),
        ]
    )   
    
    if mode == 'train':
        train_df = pd.read_csv(data_folder_dir + "/train_2.csv")
        img_list = list(map(add_img_path, train_df['Path'].values))
        label_list = list(map(label_class, train_df['Classification'].values))
        print("Train [Total]  number = ", len(img_list))
        transform_combination = train_transforms

    elif mode == 'valid':
        valid_df = pd.read_csv(data_folder_dir + "/valid.csv")
        img_list = list(map(add_img_path, valid_df['Path'].values))
        label_list = list(map(label_class, valid_df['Classification'].values))
        print("Valid [Total]  number = ", len(img_list))
        transform_combination = valid_transforms

    data_dicts   = [ {"image": image_name, "label": study_label_name, 'path':image_name} for image_name, study_label_name in zip(img_list, label_list)]

    return Dataset(data=data_dicts, transform=transform_combination), default_collate_fn

def Supervised_16Class_TEST(data_folder_dir='/home/pkg777774/child_xray/fracture/jykim'):
    valid_transforms = Compose(
        [
            # Just dicom Load
            Lambdad(keys=["image"], func=get_pixels_hu),
            Lambdad(keys=["image"], func=functools.partial(dicom_resize_and_padding_with_aspect, spatial_size=(256, 256))),                    
            Lambdad(keys=["image"], func=dicom_normalize),                    
            AddChanneld(keys=["image"]),          

            # Data Normalize               
            ToTensord(keys=["image"]),
        ]
    ) 

    test_df = pd.read_csv(data_folder_dir + "/test.csv")
    img_list = list(map(add_img_path, test_df['Path'].values))
    label_list = list(map(label_class, test_df['Classification'].values))
    print("test [Total]  number = ", len(img_list))
    transform_combination = valid_transforms

    data_dicts   = [ {"image": image_name, "label": study_label_name, 'path':image_name} for image_name, study_label_name in zip(img_list, label_list)]

    return Dataset(data=data_dicts, transform=transform_combination), default_collate_fn

    

