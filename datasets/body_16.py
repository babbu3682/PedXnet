import numpy as np
import pandas as pd
from torch.utils.data import Dataset as BaseDataset
import skimage
from pydicom import dcmread
import albumentations as albu
from PIL import Image
from datasets.datasets_utils import getItem, getItem_DownTask



label_dict = {'Knee':0,
            'Upper arm':1, 
            'Elbow':2, 
            'Hand':3, 
            'Forearm':4, 
            'Foot':5, 
            'Thigh':6,
            'Shoulder':7, 
            'Hip':8, 
            'Chest':9, 
            'Abdomen':10, 
            'Head':11,
            'Neck':12, 
            'Spine':13, 
            'Pelvis':14,
            'Lower leg':15
            }

def get_path(x):
    return x.replace('/workspace/16_chest_disease/', '/workspace/sunggu/3.Child/dataset/goto_16_disease/')

def get_label(x):
    return label_dict[x]


class Body_16_Dataset(BaseDataset):
    def __init__(self, mode='train', transform=None, num_imgs_per_cat=None, training_mode='SSL'):  
        self.mode = mode
        self.transform = transform
        self.training_mode = training_mode
        
        # 엑셀로 라벨링 할건지 다시 확인.

        df = pd.read_excel('/workspace/sunggu/3.Child/dataset/goto_16_disease/train_val_test_path_list.xlsx')

        if self.mode == 'train':
            self.img_list   = list(map(get_path,  df['train'].dropna().values))
            self.label_list = list(map(get_label, df[['train','class']].dropna()['class'].values))

        elif self.mode == 'val':
            self.img_list   = list(map(get_path,  df['val'].dropna().values))
            self.label_list = list(map(get_label, df[['val','class']].dropna()['class'].values))
            
        else :
            self.img_list   = list(map(get_path,  df['test'].dropna().values))
            self.label_list = list(map(get_label, df[['test','class']].dropna()['class'].values))   
        
        # Pydicom Error shut down
        import warnings
        warnings.filterwarnings(action='ignore') 

    def __getitem__(self, i):
        
        # Read Image
        image = dcmread(self.img_list[i]).pixel_array  # 않읽혀지는게 없지만 bias 가 생김 test 할때도 이거 써야함.    
        assert len(image.shape) == 2, print('Shape=> ', image.shape, ' 에러 파일=', self.img_list[i]) 
        target = np.array(self.label_list[i])
    
        # print("Check!", image.shape, image.dtype, image.max(), image.min())   # Check! (1982, 1982) uint16 16383 1020
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

        return getItem_DownTask(X=X, target=target, transform=self.transform, training_mode=self.training_mode)

    def __len__(self):
        return len(self.img_list)
