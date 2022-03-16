import numpy as np
import pandas as pd
from torch.utils.data import Dataset as BaseDataset
import skimage
import albumentations as albu
from PIL import Image
from datasets.datasets_utils import getItem, getItem_DownTask
from functools import partial

def get_path(mode, x):
    return "/workspace/sunggu/3.Child/dataset/RSNA_bone_age_dataset/"+mode+"/"+str(x)+".png"

def get_label(x):
    return np.float32(x)


class RSNADataset(BaseDataset):
    def __init__(self, mode='train', transform=None, num_imgs_per_cat=None, training_mode='SSL'):  
        self.mode = mode
        self.transform = transform
        self.training_mode = training_mode
        
        if self.mode == 'train':
            xray_df = pd.read_csv("/workspace/sunggu/3.Child/dataset/RSNA_bone_age_dataset/train.csv")
            self.img_list    = list(map(partial(get_path, 'train'), xray_df['id']))
            self.age_list    = list(map(get_label, xray_df['boneage']))
            self.gender_list = list(map(get_label, xray_df['male']))

        else :
            xray_df = pd.read_csv("/workspace/sunggu/3.Child/dataset/RSNA_bone_age_dataset/valid.csv")
            self.img_list    = list(map(partial(get_path, 'valid'), xray_df['id']))
            self.age_list    = list(map(get_label, xray_df['boneage']))
            self.gender_list = list(map(get_label, xray_df['male']))
        
        # Pydicom Error shut down
        import warnings
        warnings.filterwarnings(action='ignore') 

    def __getitem__(self, i):

        image_dir = self.img_list[i]
        age       = self.age_list[i]
        gender    = self.gender_list[i]

        # Read Image
        image  = np.array(Image.open(image_dir))
        age    = np.array(age)
        gender = np.array(gender)

        # print("Check!", image.shape, image.dtype, image.max(), image.min()) --> Check! (2460, 1950) uint8 255 0    
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

        return getItem_RSNA(X, gender, age, self.transform, self.training_mode)
        
    def __len__(self):
        return len(self.img_list)

