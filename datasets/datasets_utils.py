from PIL import Image
import numpy as np

def RandomRotation(X, orientation=None):
    # generate random number between 0 and n_rot to represent the rotation
    if orientation is None:
        orientation = np.random.randint(0, 4)
    
    if orientation == 0: # do nothing
        pass
    elif orientation == 1:  
        X = X.transpose(Image.ROTATE_90)
    elif orientation == 2:  
        X = X.transpose(Image.ROTATE_180)
    elif orientation == 3: 
        X = X.transpose(Image.ROTATE_270)
        
    return X, orientation


def buildLabelIndex(labels):
    label2inds = {}
    for idx, label in enumerate(labels):
        if label not in label2inds:
            label2inds[label] = []
        label2inds[label].append(idx)

    return label2inds


def getItem(X, target = None, transform=None, training_mode = 'SSL'):
    # in case of finetuning, returning the image and the target
    if training_mode == 'finetune':
        if transform is not None:
            X = transform(X)
        return X, target
        
    X1, rot1 = RandomRotation(X)
    X2, rot2 = RandomRotation(X)

    if transform is not None:
        X1 = transform(X1)
        X2 = transform(X2)
    
    return X1, rot1, X2, rot2


def getItem_DownTask(X, gender=None, target=None, transform=None, training_mode=None):
    if training_mode == 'DownTask_RSNA':
        if transform is not None:
            X = transform(X)
        return X, gender, target

    elif training_mode == 'DownTask_NIH':
        if transform is not None:
            X = transform(X)
        return X, target    

    elif training_mode == 'DownTask_Disease_16':
        if transform is not None:
            X = transform(X)
        return X, target   

    elif training_mode == 'DownTask_Frac':
        if transform is not None:
            X = transform(X)
        
    elif training_mode == '3.Ped_Pneumo':
        if transform is not None:
            X = transform(X)
        return X, target                             