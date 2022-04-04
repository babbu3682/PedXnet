import numpy as np
import pandas as pd
from torch.utils.data import Dataset as BaseDataset
import skimage
from pydicom import dcmread
import albumentations as albu
from PIL import Image
from datasets.datasets_utils import getItem, getItem_DownTask
import os
import cv2


def get_label(x):
    if '/Pneumonia/' in x:
        return 1
    else :
        return 0

class Ped_Pneumo_Dataset(BaseDataset):
    def __init__(self, mode='train', transform=None, num_imgs_per_cat=None, training_mode='SSL'):  
        self.mode = mode
        self.transform = transform
        self.training_mode = training_mode
        self.classes = ['Normal', 'Pneumonia']
        data_path = '/mnt/nas125_vol2/jeeyoungkim/v2/Pediatric_Pneumonia/dataset/split_2class'
        
        if self.mode == 'train':
            data_path = os.path.join(data_path, self.mode)
            tmp = [os.listdir(os.path.join(data_path, i)) for i in self.classes]
            tmp = [[os.path.join(data_path, self.classes[idx], j) for j in i] for idx, i in enumerate(tmp)]    
            
            self.img_list = []
            for i in tmp:
                for j in i:
                    self.img_list.append(j)
            
            self.label_list = list(map(get_label, self.img_list))

        else :
            data_path = os.path.join(data_path, self.mode)
            tmp = [os.listdir(os.path.join(data_path, i)) for i in self.classes]
            tmp = [[os.path.join(data_path, self.classes[idx], j) for j in i] for idx, i in enumerate(tmp)]    
            self.img_list = []
            for i in tmp:
                for j in i:
                    self.img_list.append(j)
            
            self.label_list = list(map(get_label, self.img_list))
            
    def __getitem__(self, i):
        
        # Read Image
        image  = cv2.imread(self.img_list[i]) # shape (1, H, W)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        target = np.array(self.label_list[i])
        
        # print("Check!", image.shape, image.dtype, image.max(), image.min())    
        if ( len(image.shape) != 2 ):
            image = image[..., 0]        

        # 전처리 1)
        image = np.clip(image, a_min=np.percentile(image, 0.5), a_max=np.percentile(image, 99.5), dtype=np.float32)

        # 전처리 3)
        image = albu.PadIfNeeded(min_height=max(image.shape), min_width=max(image.shape), always_apply=True, border_mode=0)(image=image)['image']
        image = albu.Resize(512, 512, interpolation=1, always_apply=True)(image=image)['image']

        # Normalize
        image -= image.min()
        image /= image.max() 
        image = skimage.util.img_as_ubyte(image)

        # image shape (512, 512), dtype: uint8, range:0~255

        X = Image.fromarray(image)

        return getItem_DownTask(X=X, target=target, transform=self.transform, data_set='3.Ped_Pneumo')
        
    def __len__(self):
        return len(self.img_list)









