import numpy as np
import pandas as pd
from torch.utils.data import Dataset as BaseDataset
import skimage
from pydicom import dcmread
import albumentations as albu
from PIL import Image
from datasets.datasets_utils import getItem, getItem_DownTask


def get_label(x):
    if '/fracture/' in x:
        return 1
    else :
        return 0

class Frac_Dataset(BaseDataset):
    def __init__(self, mode='train', transform=None, num_imgs_per_cat=None, training_mode='SSL'):  
        self.mode = mode
        self.transform = transform
        self.training_mode = training_mode

        if self.mode == 'train':
            self.img_list   = list_sort_nicely(glob.glob('/workspace/sunggu/3.Child/dataset/bone_fracture/train/*/*.npy'))
            self.label_list = list(map(get_label, self.img_list))

        else :
            self.img_list   = list_sort_nicely(glob.glob('/workspace/sunggu/3.Child/dataset/bone_fracture/valid/*/*.npy'))
            self.label_list = list(map(get_label, self.img_list))

    def __getitem__(self, i):
        
        # Read Image
        image  = np.load(self.img_list[i]).squeeze() # shape (1, H, W)
        target = np.array(self.label_list[i])
        
        # print("Check!", image.shape, image.dtype, image.max(), image.min())    
        if ( len(image.shape) != 2 ):
            image = image[..., 0]        

        # 전처리 1)
        image = np.clip(image, a_min=np.percentile(image, 0.5), a_max=np.percentile(image, 99.5))

        # 전처리 3)
        image = albu.PadIfNeeded(min_height=max(image.shape), min_width=max(image.shape), always_apply=True, border_mode=0)(image=image)['image']
        image = albu.Resize(512, 512, interpolation=1, always_apply=True)(image=image)['image']

        # Normalize
        image -= image.min()
        image /= image.max() 
        image = skimage.util.img_as_ubyte(image)

        # image shape (512, 512), dtype: uint8, range:0~255

        X = Image.fromarray(image)

        return getItem_DownTask(X=X, target=target, transform=self.transform, training_mode=self.training_mode)
        
    def __len__(self):
        return len(self.img_list)








