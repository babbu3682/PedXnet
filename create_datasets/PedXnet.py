import torch
import numpy as np
import pandas as pd
import functools
import cv2 
import re
import glob
import pydicom
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
    dcm_image = pydicom.read_file(path)
    image = dcm_image.pixel_array
    image = image.astype(np.int16)

    intercept = dcm_image.RescaleIntercept
    slope     = dcm_image.RescaleSlope

    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)

    image += np.int16(intercept)
    return np.array(image, dtype=np.int16)

def get_label(x):
    if '/fracture/' in x:
        return torch.tensor([1.0])
    else :
        return torch.tensor([0.0])

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
            A.Lambda(image=minmax_normalize, always_apply=True),

            # normalization
            ToTensorV2(transpose_mask=True)
        ], additional_targets={'image2':'image'})


class PedXNet_7Class_Dataset(BaseDataset):
    def __init__(self, mode="train"):
        self.mode = mode
        if mode == 'train':
            target_df = pd.read_csv("/workspace/sunggu/3.Child/PedXnet_Code_Factory/Excel_meta_data/CSV_file_zip/Sampled_Relabeling_7_class_train.csv")
            self.image_list = target_df['Path'].values
            self.label_list = target_df['Label_7'].values
        else:
            target_df = pd.read_csv("/workspace/sunggu/3.Child/PedXnet_Code_Factory/Excel_meta_data/CSV_file_zip/Sampled_Relabeling_7_class_valid.csv")
            self.image_list = target_df['Path'].values
            self.label_list = target_df['Label_7'].values
        
        self.transforms = get_transforms(mode=mode)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, i):

        # Read Image
        image = get_pixels_hu(self.image_list[i])
        
        # Read Label
        label = self.label_list[i]
        labeling = ['Chest', 'Head', 'Abdomen', 'Lower_Extremity', 'Upper_Extremity', 'Spine', 'Pelvis']
        label = torch.tensor(labeling.index(label))

        # Apply transforms
        sample = self.transforms(image=image)
        image  = sample['image']
        
        # image [B, 1, 512, 512], label [B]
        return image.float(), label.long()


class PedXNet_30Class_Dataset(BaseDataset):
    def __init__(self, mode="train"):
        self.mode = mode
        if mode == 'train':
            target_df = pd.read_csv("/workspace/sunggu/3.Child/PedXnet_Code_Factory/Excel_meta_data/CSV_file_zip/Sampled_Relabeling_30_class_train.csv")
            self.image_list = target_df['Path'].values
            self.label_list = target_df['Label_31'].values
        else:
            target_df = pd.read_csv("/workspace/sunggu/3.Child/PedXnet_Code_Factory/Excel_meta_data/CSV_file_zip/Sampled_Relabeling_30_class_valid.csv")
            self.image_list = target_df['Path'].values
            self.label_list = target_df['Label_31'].values
        
        self.transforms = get_transforms(mode=mode)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, i):

        # Read Image
        image = get_pixels_hu(self.image_list[i])
        
        # Read Label
        label = self.label_list[i]
        labeling = ['Femur', 'Mandible', 'Skull', 'Lower_Leg', 'Upper_Extremity', 'Nose',
       'Orbit', 'Finger', 'Wrist', 'Hand', 'Elbow', 'Zygomatic', 'Hip',
       'T_Spine', 'Knee', 'Shoulder', 'Lower_Extremity', 'Mastoid', 'Ankle',
       'L_Spine', 'Humerus', 'Chest', 'Whole_Spine', 'T_L_Spine', 'Cochlea',
       'Abdomen', 'C_Spine', 'Forearm', 'Toe', 'Foot']
        label = torch.tensor(labeling.index(label))

        print("Check!", image.shape, image.dtype, image.max(), image.min())

        # Apply Augmentations
        sample = self.transforms(image=image)
        image  = sample['image']
        
        # image [512, 512], label scalar
        return image.float(), label.long()


class PedXNet_68Class_Dataset(BaseDataset):
    def __init__(self, mode="train"):
        self.mode = mode
        if mode == 'train':
            target_df = pd.read_csv("/workspace/sunggu/3.Child/PedXnet_Code_Factory/Excel_meta_data/CSV_file_zip/Sampled_Relabeling_68_class_train.csv")
            self.image_list = target_df['Path'].values
            self.label_list = target_df['Label_100'].values
        else:
            target_df = pd.read_csv("/workspace/sunggu/3.Child/PedXnet_Code_Factory/Excel_meta_data/CSV_file_zip/Sampled_Relabeling_68_class_valid.csv")
            self.image_list = target_df['Path'].values
            self.label_list = target_df['Label_100'].values
        
        self.transforms = get_transforms(mode=mode)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, i):

        # Read Image
        image = get_pixels_hu(self.image_list[i])
        
        # Read Label
        label = self.label_list[i]
        labeling = ['Chest_Decubitus', 'Abdomen_Lateral', 'Mastoid', 'Knee_Lateral',
       'Upper_Extremity', 'Toe', 'Foot_Calcaneus', 'Chest_Rib', 'Finger',
       'Leg_Oblique', 'Foot_AP', 'Pevis_coccyx_sacrum', 'Orbit', 'Ankle_AP',
       'Zygomatic', 'Foot_Hindfoot', 'Pelvis_Oblique', 'Foot_Oblique',
       'Whole_Lower_AP', 'Abdomen_KUB', 'Chest_frontal', 'Shoulder_AP',
       'Knee_Oblique', 'Foot_Lateral', 'T_L_Spine', 'Forearm_Oblique',
       'Abdomen_upright', 'Hand_Oblique', 'T_Spine_AP', 'Wrist_AP',
       'Chest_Lateral', 'Pelvis_Lateral', 'Chest_Clavicle', 'Nose_Lateral',
       'Pelvis_Frogleg', 'Ankle_Lateral', 'Ankle_Stress', 'Skull_Towne',
       'Shoulder_Axial', 'Pelvis_SI_joint', 'Pelvis_Translateral',
       'Ankle_Mortise', 'Hand_PA', 'L_Spine_Lateral', 'Hand_Lateral',
       'Wrist_Oblique', 'Knee_AP', 'Skull_Lateral', 'Knee_Skyline',
       'C_spine_Lateral', 'Knee_Stress', 'Leg_AP', 'Pelvis_AP',
       'Humerus_Oblique', 'Femur_AP', 'Abdomen_supine', 'Whole_Spine_AP',
       'C_spine_Atlas', 'Cochlea', 'Mandible', 'Skull_AP', 'Nose_PNS',
       'L_Spine_Oblique', 'L_Spine_AP', 'Elbow_Lateral', 'Whole_Lower_Lateral',
       'Skull_Tangential', 'C_spine_AP']
        label = torch.tensor(labeling.index(label))

        print("Check!", image.shape, image.dtype, image.max(), image.min())

        # Apply Augmentations
        sample = self.transforms(image=image)
        image  = sample['image']
        
        # image [512, 512], label scalar
        return image.float(), label.long()