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

# Pydicom Error shut down
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
        image[image == -2000] = 0

        intercept = dcm_image.RescaleIntercept
        slope     = dcm_image.RescaleSlope

        if slope != 1:
            image = slope * image.astype(np.float64)
            image = image.astype(np.int16)

        image += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)

def dicom_normalize(image): 
    image -= image.min()
    image /= image.max() 
    return image.astype('float32')

def dicom_resize_and_padding_with_aspect(image, spatial_size):
    image = np.clip(image, a_min=np.percentile(image, 0.5), a_max=np.percentile(image, 99.5))
    image = albu.PadIfNeeded(min_height=max(image.shape), min_width=max(image.shape), always_apply=True, border_mode=0)(image=image)['image']
    image = cv2.resize(image, spatial_size, interpolation=cv2.INTER_CUBIC)
    return image

def add_img_path(x):
    return '/workspace/sunggu/3.Child/PedXnet_Code_Factory/goto_chest' + x

def minmax_normalize(image):
    image -= image.min()
    image /= image.max() 
    return image.astype('float32')

def Old_PedXNet_Dataset(mode, num_class, patch_training=False):       
    if mode == 'train':
        if num_class == 7:
            target_df = pd.read_csv("/workspace/sunggu/3.Child/Excel_meta_data/CSV_file_zip/Sampled_Relabeling_7_class_train.csv", index_col=0)
            img_list   = list(map(add_img_path, target_df['Path'].values))
            label_list = target_df['Label_7'].values

        elif num_class == 30:
            target_df = pd.read_csv("/workspace/sunggu/3.Child/Excel_meta_data/CSV_file_zip/Sampled_Relabeling_30_class_train.csv", index_col=0)
            img_list   = list(map(add_img_path, target_df['Path'].values))
            label_list = target_df['Label_31'].values

        elif num_class == 68:
            target_df = pd.read_csv("/workspace/sunggu/3.Child/Excel_meta_data/CSV_file_zip/Sampled_Relabeling_68_class_train.csv", index_col=0)
            img_list   = list(map(add_img_path, target_df['Path'].values))
            label_list = target_df['Label_100'].values

        else: 
            target_df = pd.read_csv("/workspace/sunggu/3.Child/Excel_meta_data/CSV_file_zip/Sampled_Relabeling_30_class_train.csv", index_col=0)
            img_list   = list(map(add_img_path, target_df['Path'].values))
            label_list = target_df['Label_31'].values

        data_dicts   = [{"image": image_name, "label": label_name, 'path':image_name} for image_name, label_name in zip(img_list, label_list)]        
        print("Train [Total]  number = ", len(img_list))

        if patch_training:
            transforms = Compose(
                [   
                    # Preprocessing
                    Lambdad(keys=["image"], func=get_pixels_hu),
                    Lambdad(keys=["image"], func=functools.partial(dicom_resize_and_padding_with_aspect, spatial_size=(512, 512))),                    
                    Lambdad(keys=["image"], func=dicom_normalize),                    
                    AddChanneld(keys=["image"]),   

                    # Crop  
                    RandSpatialCropSamplesd(keys=["image"], roi_size=(128, 128), num_samples=8, random_center=True, random_size=False, meta_keys=None, allow_missing_keys=False), 

                    # (45 degree rotation, vertical & horizontal flip & scaling)
                    RandFlipd(keys=["image"], prob=0.1, spatial_axis=[0, 1], allow_missing_keys=False),
                    RandRotated(keys=["image"], prob=0.1, range_x=np.pi/4, range_y=np.pi/4, range_z=0.0, keep_size=True, align_corners=False, allow_missing_keys=False),
                    RandZoomd(keys=["image"], prob=0.1, min_zoom=0.5, max_zoom=2.0, align_corners=None, keep_size=True, allow_missing_keys=False),

                    ToTensord(keys=["image"]),
                ]
            )    

        else :
            transforms = Compose(
                [
                    # Preprocessing
                    Lambdad(keys=["image"], func=get_pixels_hu),
                    Lambdad(keys=["image"], func=functools.partial(dicom_resize_and_padding_with_aspect, spatial_size=(512, 512))),                    
                    Lambdad(keys=["image"], func=dicom_normalize),                    
                    AddChanneld(keys=["image"]),              

                    # (45 degree rotation, vertical & horizontal flip & scaling)
                    RandFlipd(keys=["image"], prob=0.1, spatial_axis=[0, 1], allow_missing_keys=False),
                    RandRotated(keys=["image"], prob=0.1, range_x=np.pi/4, range_y=np.pi/4, range_z=0.0, keep_size=True, align_corners=False, allow_missing_keys=False),
                    RandZoomd(keys=["image"], prob=0.1, min_zoom=0.5, max_zoom=2.0, align_corners=None, keep_size=True, allow_missing_keys=False),

                    ToTensord(keys=["image"]),
                ]
            )              

    elif mode == 'valid':
        if num_class == 7:
            target_df  = pd.read_csv("/workspace/sunggu/3.Child/CSV_file_zip/Sampled_Relabeling_7_class_valid.csv", index_col=0)
            img_list   = list(map(add_img_path, target_df['Path'].values))
            label_list = target_df['Label_7'].values

        elif num_class == 30:
            target_df  = pd.read_csv("/workspace/sunggu/3.Child/CSV_file_zip/Sampled_Relabeling_30_class_valid.csv", index_col=0)
            target_df  = target_df.drop([1376837])
            img_list   = list(map(add_img_path, target_df['Path'].values))
            label_list = target_df['Label_31'].values

        elif num_class == 68:
            target_df  = pd.read_csv("/workspace/sunggu/3.Child/CSV_file_zip/Sampled_Relabeling_68_class_valid.csv", index_col=0)                        
            img_list   = list(map(add_img_path, target_df['Path'].values))
            label_list = target_df['Label_100'].values

        else: 
            target_df  = pd.read_csv("/workspace/sunggu/3.Child/CSV_file_zip/Sampled_Relabeling_30_class_valid.csv", index_col=0)
            target_df  = target_df.drop([1376837])
            img_list   = list(map(add_img_path, target_df['Path'].values))
            label_list = target_df['Label_31'].values

        data_dicts   = [{"image": image_name, "label": label_name} for image_name, label_name in zip(img_list, label_list)]        
        print("Valid [Total]  number = ", len(img_list))

        transforms = Compose(
            [
                # Preprocessing
                Lambdad(keys=["image"], func=get_pixels_hu),
                Lambdad(keys=["image"], func=functools.partial(dicom_resize_and_padding_with_aspect, spatial_size=(512, 512))),                    
                Lambdad(keys=["image"], func=dicom_normalize),                    
                AddChanneld(keys=["image"]),          

                # (45 degree rotation, vertical & horizontal flip & scaling)
                RandFlipd(keys=["image"], prob=0.1, spatial_axis=[0, 1], allow_missing_keys=False),
                RandRotated(keys=["image"], prob=0.1, range_x=np.pi/4, range_y=np.pi/4, range_z=0.0, keep_size=True, align_corners=False, allow_missing_keys=False),
                RandZoomd(keys=["image"], prob=0.1, min_zoom=0.5, max_zoom=2.0, align_corners=None, keep_size=True, allow_missing_keys=False),

                ToTensord(keys=["image"]),
            ]
        )              
  
    else: 
        if num_class == 7:
            target_df  = pd.read_csv("/workspace/sunggu/3.Child/CSV_file_zip/Sampled_Relabeling_7_class_test.csv", index_col=0)
            img_list   = list(map(add_img_path, target_df['Path'].values))
            label_list = target_df['Label_7'].values

        elif num_class == 30:
            target_df  = pd.read_csv("/workspace/sunggu/3.Child/CSV_file_zip/Sampled_Relabeling_30_class_test.csv", index_col=0)
            target_df  = target_df.drop([1376837])
            img_list   = list(map(add_img_path, target_df['Path'].values))
            label_list = target_df['Label_31'].values

        elif num_class == 68:
            target_df  = pd.read_csv("/workspace/sunggu/3.Child/CSV_file_zip/Sampled_Relabeling_68_class_test.csv", index_col=0)                        
            img_list   = list(map(add_img_path, target_df['Path'].values))
            label_list = target_df['Label_100'].values

        else: 
            target_df  = pd.read_csv("/workspace/sunggu/3.Child/CSV_file_zip/Sampled_Relabeling_30_class_test.csv", index_col=0)
            img_list   = list(map(add_img_path, target_df['Path'].values))
            label_list = target_df['Label_31'].values

        data_dicts   = [{"image": image_name, "label": label_name} for image_name, label_name in zip(img_list, label_list)]        
        print("TEST [Total]  number = ", len(img_list))

        transforms = Compose(
            [
                # Preprocessing
                Lambdad(keys=["image"], func=get_pixels_hu),
                Lambdad(keys=["image"], func=functools.partial(dicom_resize_and_padding_with_aspect, spatial_size=(512, 512))),                    
                Lambdad(keys=["image"], func=dicom_normalize),                    
                AddChanneld(keys=["image"]),          

                # (45 degree rotation, vertical & horizontal flip & scaling)
                RandFlipd(keys=["image"], prob=0.1, spatial_axis=[0, 1], allow_missing_keys=False),
                RandRotated(keys=["image"], prob=0.1, range_x=np.pi/4, range_y=np.pi/4, range_z=0.0, keep_size=True, align_corners=False, allow_missing_keys=False),
                RandZoomd(keys=["image"], prob=0.1, min_zoom=0.5, max_zoom=2.0, align_corners=None, keep_size=True, allow_missing_keys=False),

                ToTensord(keys=["image"]),
            ]
        )  

    return Dataset(data=data_dicts, transform=transforms)





def New_PedXNet_Dataset(mode):  
    # df = pd.read_csv('/workspace/sunggu/3.Child/dataset/Upstream_PedXnet_final.csv', index_col=0, low_memory=False).sample(n=4000, random_state=7)
    df = pd.read_csv('/workspace/sunggu/3.Child/dataset/Upstream_PedXnet_final.csv', index_col=0, low_memory=False)

    train_transforms = Compose(
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
            
            # Data Normalize
            Lambdad(keys=["image"], func=minmax_normalize),                    
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
            # Lambdad(keys=["image"], func=minmax_normalize),    # it is the same as the 'docom_normalize'                
            ToTensord(keys=["image"]),
        ]
    )         

    if mode == 'train':
        path_img_list     = df[df['Mode'] == 'train']['Path'].values
        study_label_list  = df[df['Mode'] == 'train']['Study_Description'].values
        series_label_list = df[df['Mode'] == 'train']['Series_Description'].values
        print("Train [Total]  number = ", len(path_img_list))    
        transform_combination = train_transforms

    elif mode == 'valid':
        path_img_list     = df[df['Mode'] == 'valid']['Path'].values
        study_label_list  = df[df['Mode'] == 'valid']['Study_Description'].values
        series_label_list = df[df['Mode'] == 'valid']['Series_Description'].values
        print("Valid [Total]  number = ", len(path_img_list))    
        transform_combination = valid_transforms
  
    else: 
        path_img_list     = df[df['Mode'] == 'test']['Path'].values
        study_label_list  = df[df['Mode'] == 'test']['Study_Description'].values
        series_label_list = df[df['Mode'] == 'test']['Series_Description'].values
        print("Test [Total]  number = ", len(path_img_list))    
        transform_combination = valid_transforms

    # data_dicts   = [{"image": image_name, "study_label": study_label_name, "series_label": series_label_name, 'path':image_name} for image_name, study_label_name, series_label_name in zip(path_img_list, study_label_list, series_label_list)]        
    data_dicts   = [ {"image": image_name, "study_label": study_label_name, 'path':image_name} for image_name, study_label_name in zip(path_img_list, study_label_list) ]        
    # data_dicts   = [ {"image": image_name, 'path':image_name} for image_name in path_img_list ]        

    return Dataset(data=data_dicts, transform=transform_combination)








# # # Read Image
# import pydicom
# import albumentations as albu
# def preprocessing(x):
#     image = pydicom.dcmread(x).pixel_array  # 않읽혀지는게 없지만 bias 가 생김 test 할때도 이거 써야함.    
    
#     assert len(image.shape) == 2, print('에러 파일=', image_dir)

#     # 전처리 1)
#     image = np.clip(image, a_min=np.percentile(image, 0.5), a_max=np.percentile(image, 99.5))

#     # 전처리 3)
#     image = albu.PadIfNeeded(min_height=max(image.shape), min_width=max(image.shape), always_apply=True, border_mode=0)(image=image)['image']
#     image = albu.Resize(512, 512, interpolation=1, always_apply=True)(image=image)['image']

#     # 전처리 2)
#     image -= image.min()
#     image /= image.max() 
#     image = skimage.util.img_as_ubyte(image)

#     # # image shape (512, 512), dtype: uint8, range:0~255

#     return x 




