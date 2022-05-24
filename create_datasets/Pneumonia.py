import functools
import cv2
import glob
import torch
import skimage
import numpy as np
import re
import albumentations as A
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


def get_jpeg(path):
    image = cv2.imread(path, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype('float32')

    return image


def get_label(x):
    if '/Normal/' in x:
        return 0
    else :
        return 1


def resize_and_padding_with_aspect(image, spatial_size):
    image = np.clip(image, a_min=np.percentile(image, 0.5), a_max=np.percentile(image, 99.5))

    image = A.PadIfNeeded(min_height=max(image.shape), min_width=max(image.shape), always_apply=True, border_mode=0)(image=image)['image']
    image = cv2.resize(image, spatial_size, interpolation=cv2.INTER_CUBIC)
    
    return image


def Albu_2D_Transform_Compose(input):
    image = input['image'].squeeze(0)

    Trans = A.Compose([
                A.OneOf([
                    A.MedianBlur(blur_limit=3, p=0.1),
                    A.MotionBlur(p=0.2),
                    A.Sharpen(alpha=(0.01, 0.2), lightness=(0.5, 1.0), always_apply=False, p=0.2)
                    ], p=0.2),
                A.RandomBrightnessContrast(brightness_limit=0, contrast_limit=0.2, p=0.6),
                A.OneOf([
                    A.GaussNoise(var_limit=0.01, p=0.2),
                    A.MultiplicativeNoise(p=0.2),
                    ], p=0.2),
                A.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=0,
                                        val_shift_limit=0.1, p=0.3),
                A.ShiftScaleRotate(shift_limit=0.062,
                                    scale_limit=0.2,
                                    rotate_limit=5, p=0.2),
                A.OneOf([
                    A.OpticalDistortion(distort_limit=0.05, p=0.3),
                    ], p=0.2)
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


def Pneumonia_Dataset(mode, data_folder_dir="/mnt/nas125_vol2/kanggilpark/child/jyp_child/data"):
    train_transforms = Compose(
        [
            # Image Load
            Lambdad(keys=['image'], func=get_jpeg),
            Lambdad(keys=['image'], func=functools.partial(resize_and_padding_with_aspect, spatial_size=(512, 512))),
            Lambdad(keys=['image'], func=functools.partial(minmax_normalize, option=False)),
            AddChanneld(keys=['image']),

            # Additinal Augmentation
            Albu_2D_Transform_Compose,

            # Normalize
            Lambdad(keys=['image'], func=functools.partial(minmax_normalize, option=False)),
            ToTensord(keys=['image']),

        ]
    )
    valid_transforms = Compose(
        [
            # Image Load
            Lambdad(keys=['image'], func=get_jpeg),
            Lambdad(keys=['image'], func=functools.partial(resize_and_padding_with_aspect, spatial_size=(512, 512))),
            Lambdad(keys=['image'], func=functools.partial(minmax_normalize, option=False)),
            AddChanneld(keys=['image']),

            ToTensord(keys=['image']),
        ]
    )
    if mode == 'train':
        img_list     = list_sort_nicely(glob.glob(data_folder_dir + "/train/*/*.jpeg"))
        label_list   = [get_label(i) for i in list_sort_nicely(glob.glob(data_folder_dir + "/train/*/*.jpeg"))]
        print("Train [Total]  number = ", len(img_list))
        transform_combination = train_transforms

    elif mode == 'valid':
        img_list     = list_sort_nicely(glob.glob(data_folder_dir + "/valid/*/*.jpeg"))
        label_list   = [get_label(i) for i in list_sort_nicely(glob.glob(data_folder_dir + "/valid/*/*.jpeg"))]
        print("Valid [Total]  number = ", len(img_list))
        transform_combination = valid_transforms

    data_dicts   = [{"image": image_name, "image_path": image_name, "label": label_name} for image_name, label_name in zip(img_list, label_list)]
    
    return Dataset(data=data_dicts, transform=transform_combination), default_collate_fn


def Pneumonia_Dataset_TEST(data_folder_dir="/mnt/nas125_vol2/kanggilpark/child/data"):
    valid_transforms = Compose(
        [
            # Image Load
            Lambdad(keys=['image'], func=get_jpeg),
            Lambdad(keys=['image'], func=functools.partial(resize_and_padding_with_aspect, spatial_size=(512, 512))),
            Lambdad(keys=['image'], func=functools.partial(minmax_normalize, option=False)),
            AddChanneld(keys=['image']),

            ToTensord(keys=['image']),
        ]
    )
    img_list     = list_sort_nicely(glob.glob(data_folder_dir + "/test/*/*.jpeg"))
    label_list   = [get_label(i) for i in list_sort_nicely(glob.glob(data_folder_dir + "/test/*/*.jpeg"))]
    print("Test [Total]  number = ", len(img_list))
    transform_combination = valid_transforms

    data_dicts   = [{"image": image_name, "image_path": image_name, "label": label_name} for image_name, label_name in zip(img_list, label_list)]

    return Dataset(data=data_dicts, transform=transform_combination), default_collate_fn