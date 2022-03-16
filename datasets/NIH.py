import numpy as np
import pandas as pd
from torch.utils.data import Dataset as BaseDataset
import skimage
from pydicom import dcmread
import albumentations as albu
from PIL import Image
from datasets.datasets_utils import getItem, getItem_DownTask


def get_path(x):
    return "/workspace/sunggu/3.Child/dataset/NIH_chest_dataset/images/"+x.split(" ")[0]

def get_label(x):
    return list(map(np.float32, x.replace("\n", "").split(" ")[1:]))


class NIHDataset(BaseDataset):
    def __init__(self, mode='train', transform=None, num_imgs_per_cat=None, training_mode='SSL'):  
        self.mode = mode
        self.transform = transform
        self.training_mode = training_mode

        self.label_dict = {'Atelectasis':0, 'Cardiomegaly':1,  'Effusion':2, 'Infiltration':3, 'Mass':4, 'Nodule':5, 
                    'Pneumonia':6, 'Pneumothorax':7, 'Consolidation':8, 'Edema':9, 'Emphysema':10, 
                    'Fibrosis': 11, 'Pleural_Thickening': 12, 'Hernia': 13}    

        if self.mode == 'train':
            train_list       = open("/workspace/sunggu/3.Child/dataset/NIH_chest_dataset/NIH_train_list.txt", 'r').readlines()
            self.img_list   = list(map(get_path, train_list))
            self.label_list = list(map(get_label, train_list))
        else :
            valid_list       = open("/workspace/sunggu/3.Child/dataset/NIH_chest_dataset/NIH_val_list.txt", 'r').readlines()
            self.img_list   = list(map(get_path, valid_list))
            self.label_list = list(map(get_label, valid_list))
        
        # Pydicom Error shut down
        import warnings
        warnings.filterwarnings(action='ignore') 

    def __getitem__(self, i):
        
        # print("경로 = ", self.img_list[i])
        # Read Image
        image  = np.array(Image.open(self.img_list[i]))
        target = np.array(self.label_list[i])
        
        # print("Check!", image.shape, image.dtype, image.max(), image.min())    
        if ( len(image.shape) != 2 ):
            image = image[..., 0]        

        # 전처리 1)
        image = np.clip(image, a_min=np.percentile(image, 0.5), a_max=np.percentile(image, 99.5))

        # 전처리 3)
        image = albu.PadIfNeeded(min_height=max(image.shape), min_width=max(image.shape), always_apply=True, border_mode=0)(image=image)['image']
        image = albu.Resize(512, 512, interpolation=1, always_apply=True)(image=image)['image']

        # 전처리 2)
        image -= image.min()
        image /= image.max() 
        image = skimage.util.img_as_ubyte(image)

        # image shape (512, 512), dtype: uint8, range:0~255

        X = Image.fromarray(image)

        return getItem_NIH(X, target, self.transform, self.training_mode)
        
    def __len__(self):
        return len(self.img_list)









