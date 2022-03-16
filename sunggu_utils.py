import numpy as np
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import matplotlib
import re
import os
import SimpleITK as sitk
import ipywidgets as widgets
import torch

######### 리스트 정렬 기능 ##########
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

###### 값 확인 할때 사용 #########
def check_value(name, x):
    print(str(name) + ".shape = ", x.shape)
    print(str(name) + ".dtype = ", x.dtype)
    print(str(name) + ".max = ", x.max())
    print(str(name) + ".min = ", x.min())

    value, counts = np.unique(x, return_counts=True)
    # 5개만
    if (len(value) > 5):
        print(str(name) + ".unique 5 values = ", value[:5])
        print(str(name) + ".unique 5 counts = ", counts[:5])
        
    else : 
        print(str(name) + ".unique less 5 values = ", value[:len(value)])
        print(str(name) + ".unique less 5 counts = ", counts[:len(counts)])



########### 시각화 함수 ################
def plot_auc_roc(label,pred):
    """
    Example
    =======================================
    y_true = np.random.randint(0,2,(100,1))
    y_pred = np.random.rand(100,1)
    print(y_true.shape,y_pred.shape)
    plot_auc_roc(y_true,y_pred)
    """
    ground_truth_labels =label.ravel()
    score_value = pred.ravel()
    fpr, tpr, _ = roc_curve(ground_truth_labels,score_value)
    roc_auc = auc(fpr,tpr)
    fig, ax = plt.subplots(1,1)
    ax.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic')
    ax.legend(loc="lower right")
    return fig

def plot_confusion_matrix(y_true, y_pred, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    Example
    =======================================
    y_true = np.random.randint(0,5,(100,1))
    y_pred = np.random.randint(0,5,(100,1))
    classes = ['1','2','3','4','5']
    print(np.unique(y_true),np.unique(y_pred))
    print(y_true.shape,y_pred.shape)
    plot_confusion_matrix(y_true, y_pred, classes, title='Test confusion matrix')
    """
    font = {'weight' : 'normal','size' : 16}
    matplotlib.rc('font', **font)
    cm = confusion_matrix(y_true, y_pred)
    cm = np.transpose(cm)
    cm = np.flipud(cm)
    cm_origin = cm.copy()
    cm = cm.astype('float') / cm.sum(axis=0)[np.newaxis]

    fig, ax = plt.subplots(1,1,figsize=(5,5))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    fmt = '.2f'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j,i, format(cm[i, j], fmt)+'\n({})'.format(cm_origin[i,j]),
                  horizontalalignment="center",
                  verticalalignment="center",
                  color="white" if cm[i, j] > thresh else "black")

    cm = np.flipud(cm)
    cm_origin = np.flipud(cm_origin)

    acc = accuracy_score(y_true, y_pred)
    specificity = cm_origin[0,0]/(cm_origin[0,0]+cm_origin[1,0])
    sensitivity = cm_origin[1,1]/(cm_origin[0,1]+cm_origin[1,1])
    
    plt.title(title+'\n'+'Accuracy : '+str(acc)[:6]+'    Sensitive : '+str(sensitivity)[:6]+'    Specific : '+str(specificity)[:6]+'\n',{'fontsize': 14})
    plt.xlabel('Ground Truth', fontsize=16, labelpad=14)
    plt.ylabel('Predict', fontsize=16, labelpad=14)
    
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    im.set_clim(0,1)
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    tick_marks = np.arange(len(classes))
    plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
    plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True
    plt.xticks(tick_marks, classes, ha='center', fontsize=14)
    plt.yticks(tick_marks, classes[::-1], va='center', fontsize=14)
    plt.gca().invert_yaxis()
    plt.show()
#     return fig

def plot3_row(**images):
    """
    example)

    plot_row(image = real, gt_mask = gt_mask, prediction = pr_mask)
    """
    n = len(images)
    plt.figure(figsize=(16, 5))

    plt.subplot(1, n, 1)
    plt.xticks([])
    plt.yticks([])
    plt.title('plot 1'.title())
    plt.imshow(images['image'], 'gray')
    
    plt.subplot(1, n, 2)
    plt.xticks([])
    plt.yticks([])
    plt.title('plot 2'.title())
    plt.imshow(images['image'], 'gray')
    plt.imshow(images['gt_mask'], alpha=0.3)
    
    plt.subplot(1, n, 3)
    plt.xticks([])
    plt.yticks([])
    plt.title('plot 3'.title())
    plt.imshow(images['image'], 'gray')
    plt.imshow(images['prediction'], alpha=0.3)

    plt.show()



class MultiImageDisplay(object):
    def __init__(self, image_list, axis=0, shared_slider=False, title_list=None, window_level_list= None, figure_size=(10,8), horizontal=True):
        self.get_window_level_numpy_array(image_list, window_level_list)
        if title_list:
            if len(image_list)!=len(title_list):
                raise ValueError('Title list and image list lengths do not match')
            self.title_list = list(title_list)
        else:
            self.title_list = ['']*len(image_list)

        # Our dynamic slice, based on the axis the user specifies
        self.slc = [slice(None)]*3
        self.axis = axis

        # Create a figure.
        col_num, row_num = (len(image_list), 1)  if horizontal else (1, len(image_list))
        self.fig, self.axes = plt.subplots(row_num,col_num,figsize=figure_size)
        if len(image_list)==1:
            self.axes = [self.axes]

        ui = self.create_ui(shared_slider)
        # Display the data and the controls, first time we display the image is outside the "update_display" method
        # as that method relies on the previous zoom factor which doesn't exist yet.
        for ax, npa, slider, min_intensity, max_intensity in zip(self.axes, self.npa_list, self.slider_list, self.min_intensity_list, self.max_intensity_list):
            self.slc[self.axis] = slice(slider.value, slider.value+1)
            # Need to use squeeze to collapse degenerate dimension (e.g. RGB image size 124 124 1 3)
            ax.imshow(np.squeeze(npa[self.slc]),
                      cmap=plt.cm.Greys_r,
                      vmin=min_intensity,
                      vmax=max_intensity)
        self.update_display()
        display(ui)

    def create_ui(self, shared_slider):
        # Create the active UI components. Height and width are specified in 'em' units. This is
        # a html size specification, size relative to current font size.
        ui = None

        if shared_slider:
            # Validate that all the images have the same size along the axis which we scroll through
            sz = self.npa_list[0].shape[self.axis]
            for npa in self.npa_list:
                       if npa.shape[self.axis]!=sz:
                           raise ValueError('Not all images have the same size along the specified axis, cannot share slider.')

            slider = widgets.IntSlider(description='image slice:',
                                      min=0,
                                      max=sz-1,
                                      step=1,
                                      value = int((sz-1)/2),
                                      width='20em')
            slider.observe(self.on_slice_slider_value_change, names='value')
            self.slider_list = [slider]*len(self.npa_list)
            ui = widgets.Box(padding=7, children=[slider])
        else:
            self.slider_list = []
            for npa in self.npa_list:
                slider = widgets.IntSlider(description='image slice:',
                                           min=0,
                                           max=npa.shape[self.axis]-1,
                                           step=1,
                                           value = int((npa.shape[self.axis]-1)/2),
                                           width='20em')
                slider.observe(self.on_slice_slider_value_change, names='value')
                self.slider_list.append(slider)
            ui = widgets.Box(padding=7, children=self.slider_list)
        return ui

    def get_window_level_numpy_array(self, image_list, window_level_list):
        # Using GetArray and not GetArrayView because we don't keep references
        # to the original images. If they are deleted outside the view would become
        # invalid, so we use a copy wich guarentees that the gui is consistent.
        self.npa_list = list(map(sitk.GetArrayFromImage, image_list))
        if not window_level_list:
            self.min_intensity_list = list(map(np.min, self.npa_list))
            self.max_intensity_list = list(map(np.max, self.npa_list))
        else:
            self.min_intensity_list = list(map(lambda x: x[1]-x[0]/2.0, window_level_list))
            self.max_intensity_list = list(map(lambda x: x[1]+x[0]/2.0, window_level_list))

    def on_slice_slider_value_change(self, change):
        self.update_display()

    def update_display(self):

        # Draw the image(s)
        for ax, npa, title, slider, min_intensity, max_intensity in zip(self.axes, self.npa_list, self.title_list, self.slider_list, self.min_intensity_list, self.max_intensity_list):
            # We want to keep the zoom factor which was set prior to display, so we log it before
            # clearing the axes.
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()

            self.slc[self.axis] = slice(slider.value, slider.value+1)
            ax.clear()
            # Need to use squeeze to collapse degenerate dimension (e.g. RGB image size 124 124 1 3)
            ax.imshow(np.squeeze(npa[self.slc]),
                      cmap=plt.cm.Greys_r,
                      vmin=min_intensity,
                      vmax=max_intensity)
                      
            ax.set_title(title)
            ax.set_axis_off()

            # Set the zoom factor back to what it was before we cleared the axes, and rendered our data.
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

        self.fig.canvas.draw_idle()

def multi_image_display2D(image_list, title_list=None, window_level_list= None, figure_size=(10,8), horizontal=True):

    if title_list:
        if len(image_list)!=len(title_list):
            raise ValueError('Title list and image list lengths do not match')
    else:
        title_list = ['']*len(image_list)

    # Create a figure.
    col_num, row_num = (len(image_list), 1)  if horizontal else (1, len(image_list))
    fig, axes = plt.subplots(row_num, col_num, figsize=figure_size)
    if len(image_list)==1:
        axes = [axes]

    # Get images as numpy arrays for display and the window level settings
    npa_list = list(map(sitk.GetArrayViewFromImage, image_list))
    if not window_level_list:
        min_intensity_list = list(map(np.min, npa_list))
        max_intensity_list = list(map(np.max, npa_list))
    else:
        min_intensity_list = list(map(lambda x: x[1]-x[0]/2.0, window_level_list))
        max_intensity_list = list(map(lambda x: x[1]+x[0]/2.0, window_level_list))

    # Draw the image(s)
    for ax, npa, title, min_intensity, max_intensity in zip(axes, npa_list, title_list, min_intensity_list, max_intensity_list):
        ax.imshow(npa,
                  cmap=plt.cm.Greys_r,
                  vmin=min_intensity,
                  vmax=max_intensity)
        ax.set_title(title)
        ax.set_axis_off()
    fig.tight_layout()


def plot_3D(images, images_title_list, fig_size=(8,4) ,wl_list=None):
    import SimpleITK as sitk
    """
    %matplotlib notebook <-- Need to launch this first!
    Interactive display single channel 3D or batched image like CT and MRI!
    
    Input Parameter
    - images : numpy images [image1,image2,...] 넘파이
    - images_title_list : string ['image1','image2',...]
    - fig_size : same as matplotlib.pyplot.figure(fig_size())
    - wl_list : TODO... window_level_adjust
    ex) plot_3D([image, image],['image','target'], fig_size=(8, 4))
    """         
    for idx in range(len(images)):
        image_ = images[idx]
        image_ = sitk.GetImageFromArray(image_)
        images[idx] = image_
        
    if images[0].GetDimension()==2:
        multi_image_display2D(image_list=images, figure_size=fig_size, window_level_list=wl_list)
    else:
        MultiImageDisplay(image_list=images, title_list=images_title_list, figure_size=fig_size, window_level_list=wl_list,shared_slider=True)



def take_list(images_dir):
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

    ids = os.listdir(images_dir)  # train 시 "/data/train" -> 폴더명 리스트 쫙        
    images_fps = [os.path.join(images_dir, image_id) for image_id in ids]  # 1개의 png파일의 주소 리스트 목록
    images_fps = list_sort_nicely(images_fps)

    return images_fps

# 때론 glob이 더 편할때도 있다.
# import glob
# glob.glob('/workspace/sunggu/nnUNet/nnUNet_raw_data_base/nnUNet_cropped_data/Task999_Hemo/*.npz')


def make_binary(input_mask, threshold=0.5):
    mask = input_mask.copy()
    mask[mask < threshold]= 0
    mask[mask >= threshold]= 1
    return mask
    

def extract_label_mask(input_mask, target_value=1):
    mask = input_mask.copy()
    mask[mask != target_value]= 0
    mask[mask == target_value]= 1
    return mask

def find_dir(path_dir):
    # '''
    # path_dir 앞에 r 붙이기!! 
    # ex) r""\\192.168.45.100\forGPU\sunggu\nnUNet\nnUNet_preprocessed\Task004_Hippocampus\nnUNetData_plans_v2.1_2D_stage0\hippocampus_001.npz"
    # '''
    file_dir = '/workspace/'+('/').join( path_dir.split('\\')[4:] )
    print(file_dir)
    return file_dir


# def topk_index():
#     best_list = []
#     for a,b,c in zip(D,D1,D2):
#         best_list.append(a+b+c)

#     # top 5 index 추출!
#     print("Top3 index = ", np.array(best_list).argsort()[-3:][::-1]) # [::-1] -> 인덱스 거꾸로 (=reverse)  # 인덱스를 주는것!

'''
        plt.cla() 
        plt.clf()
        plt.close('all')
'''

# '/'.join( b.split('/')[:-1] )


# np.array_equal(a,b)

# x[startAt:endBefore:skip]

# 새폴더 만들기
# if not os.path.exists(png_folder_dir):
#     os.makedirs(png_folder_dir, mode=0o777)

# os.makedirs(pathm, mode=0o777, exist_ok=True)

# 리스트 컴프리헨션


# # save
# import pickle
# with open('/workspace/sunggu/소아/dicom_header_91547.pickle', 'wb') as f:
#     pickle.dump(dicom_header, f)

# # load
# import pickle
# with open('/workspace/sunggu/소아/pure_list_91547.pickle', 'rb') as f:
#     pure_list = pickle.load(f)

# for m in module.modules():

# 새폴더 만들기

# def make_folder(path_dir):
#     if not os.path.exists(path_dir):
#         os.makedirs(path_dir, mode=0o777)
# os.makedirs(pathm, mode=0o777, exist_ok=True)


# import sys
# sys.path.append(os.path.abspath('/workspace/sunggu'))
# from sunggu_utils import check_value, take_list, plot_confusion_matrix, list_sort_nicely, find_dir




# 그림 그리는 tool default

# plt.figure(figsize=(18,10))
# plt.title('')

# plt.xlabel('Ground Truth')
# plt.ylabel('Predicted')

# plt.subplot(141)
# plt.imshow( label_seg )

# plt.subplot(142)
# plt.imshow( label_seg )
# plt.xticks(tick_marks, classes, ha='center')
# plt.yticks(tick_marks, classes[::-1], va='center')
# plt.xlim(2000, 2050)
# plt.ylim(2000, 2050)

# plt.show()

# plt.cla() 
# plt.clf()
# plt.close('all')





# import torch.nn as nn
# class WS_Conv2d(nn.Conv2d):

#     def __init__(self, in_channels, out_channels, kernel_size, stride=1,
#                  padding=0, dilation=1, groups=1, bias=True):
#         super(WS_Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
#                  padding, dilation, groups, bias)

#     def forward(self, x):
#         weight = self.weight
#         weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
#         weight = weight - weight_mean
#         std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
#         weight = weight / std.expand_as(weight)
#         return F.conv2d(x, weight, self.bias, self.stride,
#                         self.padding, self.dilation, self.groups)


# def check_res(x):
#     if ( len(x._modules) == 0 ):
#         return True
    
# def chance_bn(model):
#     for module_1 in model._modules:
# #         print("#", module_1)
#         if ( check_res(model._modules[module_1]) ):
#             if (type(model._modules[module_1]) == torch.nn.BatchNorm2d):
#                 model._modules[module_1] = torch.nn.GroupNorm(num_channels=model._modules[module_1].num_features, num_groups=16)
    
#     for ch in model.children():  
#         chance_bn(ch)

        
# def chance_conv2d(model):
#     for module_1 in model._modules:
# #         print("#", module_1)
#         if ( check_res(model._modules[module_1]) ):
#             if (type(model._modules[module_1]) == torch.nn.Conv2d):
#                 model._modules[module_1] = WS_Conv2d(in_channels = model._modules[module_1].in_channels,
#                                                      out_channels = model._modules[module_1].out_channels,
#                                                      kernel_size = model._modules[module_1].kernel_size,
#                                                      stride = model._modules[module_1].stride,
#                                                      padding = model._modules[module_1].padding,
#                                                      dilation = model._modules[module_1].dilation, 
#                                                      groups = model._modules[module_1].groups)
    
#     for ch in model.children():  
#         chance_conv2d(ch)




# import albumentations as albu
# lena = cv2.imread('/workspace/sunggu/Public_dataset/lena.png')
# # Aug_list = [                 albu.IAASharpen(),
# #                  albu.Blur(),
# #                  albu.MotionBlur(),]

# for aug in Trans:
#     plt.figure(figsize=(10,10))
#     plt.subplot(121)
#     plt.title('Before')
#     plt.imshow(lena)

#     plt.subplot(122)
#     plt.title('After')
#     after = aug(image=lena)['image']
#     plt.imshow(after)

#     plt.show()



def change_value_for_visualize(target):
    empty = torch.zeros(target.shape)
    stack = torch.stack([empty, empty, empty], axis=-1)
    stack[..., 0][target == 0] = 0
    stack[..., 1][target == 0] = 0
    stack[..., 2][target == 0] = 0

    # 레드 Pancreas
    stack[..., 0][target == 1] = 1
    stack[..., 1][target == 1] = 0
    stack[..., 2][target == 1] = 0

    # 그린 Cyst
    stack[..., 0][target == 2] = 0
    stack[..., 1][target == 2] = 1
    stack[..., 2][target == 2] = 0

    return stack


def diagnose_network(net, print_full_max=False, print_full_min=False, k=3):
    """Calculate and print the mean of average absolute(gradients)"""
    max = []
    min = []
    median = []
    mean = []
    name_list=[]
    
    data = {}

    for name, param in net.named_parameters():
        if (param.grad is not None) and ("bias" not in name):
            name_list.append(name)
            mean.append(torch.mean(torch.abs(param.grad.data)).item())
            max.append(torch.max(torch.abs(param.grad.data)).item())
            min.append(torch.min(torch.abs(param.grad.data)).item())
            median.append(torch.median(torch.abs(param.grad.data)).item())
        
    
    print("====Gradient Mean 값 체크====")
    print("Grad layer 갯수 = ", len(name_list))
    print("Mean   =", np.mean(mean))
    print("Max    =", np.mean(max))
    print("Min    =", np.mean(min))
    print("Median =", np.mean(median))
    
    if (print_full_max):
        for i in np.array(max).argsort()[::-1]:
            print(name_list[i], "[", max[i], "]")
    
    elif (print_full_min):
        for i in np.array(min).argsort():
            print(name_list[i], "[", min[i], "]")
        
    else :
        print('*불안정한거 k='+str(k)+'개 뽑기')
        print('[Exploding...]')
        for i in np.array(max).argsort()[-k:][::-1]:
            print(name_list[i], "[", max[i], "]")
        print('[Vanishing...]')    
        for i in np.array(max).argsort()[:k]:
            print(name_list[i], "[", min[i], "]")

import json
def check_grad_network(net, save_path):
    """Calculate and print the mean of average absolute(gradients)"""
    max = []
    min = []
    median = []
    mean = []
    net_grad = {}
    for name, param in net.named_parameters():
        if (param.grad is not None) and ("bias" not in name):
            net_grad[name]={}
            net_grad[name]['mean']=torch.mean(torch.abs(param.grad.data)).item()
            net_grad[name]['max']=torch.max(torch.abs(param.grad.data)).item()
            net_grad[name]['min']=torch.min(torch.abs(param.grad.data)).item()
            net_grad[name]['median']=torch.median(torch.abs(param.grad.data)).item()


    with open(save_path, 'w') as outfile:
        json.dump(net_grad, outfile, indent="\t")



# # 특정 단어+숫자 찾기
# for i in range(10):
#     exec("words = words.split('_F' + str(i))[0]")
#     exec("words = words.split('_M' + str(i))[0]")




# import albumentations as albu
# lena = cv2.imread('/workspace/sunggu/Public_dataset/lena.png')
# # Aug_list = [                 albu.IAASharpen(),
# #                  albu.Blur(),
# #                  albu.MotionBlur(),]

# for aug in Trans:
#     plt.figure(figsize=(10,10))
#     plt.subplot(121)
#     plt.title('Before')
#     plt.imshow(check[..., 100])

#     plt.subplot(122)
#     plt.title('After')
#     after = aug(image=check[..., 100])['image']
#     plt.imshow(after)

#     plt.show()



# # Onehot encoding
# labels = [(test_labels.item() == v) for v in [0,1,2,3,4,5,6,7]]
# mask = np.stack(labels, axis=-1).astype('float')



import torch
import torchvision
import monai

# 32,512,512 형태 shape 이어야한다.

# def plot_grid_3d(figsize, np_img, np_mask, np_pred, save=None):
#     %matplotlib inline
#     # np_img = [batch, 512, 512] 형태 
#     # np_mask = [batch, 512, 512] 형태 
    
#     # Windowing for Visualize
#     np_img = monai.transforms.ScaleIntensityRange(a_min=-100.0, a_max=200.0, b_min=0.0, b_max=1.0, clip=True)(np_img)
    
#     torch_img = torch.from_numpy(np_img.astype('float')).unsqueeze(1)
#     torch_mask = torch.from_numpy(np_mask.astype('float')).unsqueeze(1)
#     torch_pred = torch.from_numpy(np_pred.astype('float')).unsqueeze(1)
    
    
#     # [batch(=tile 만들 여러장), channel, 512, 512] 형태 꼭 지켜줘야함
#     tiled_img  = torchvision.utils.make_grid(torch_img, 
#                                               nrow = 3, # 순서대로 옆으로 5장씩 0,1,2,3,5 다음줄 6,7,8,9,10 이런식
#                                               normalize = True,
#                                               padding = 3,
#                                               range = None,
#                                               scale_each = True,
#                                               pad_value = 1.0)[0] 
    
#     tiled_mask  = torchvision.utils.make_grid(torch_mask, 
#                                               nrow = 3, # 순서대로 옆으로 5장씩 0,1,2,3,5 다음줄 6,7,8,9,10 이런식
#                                               normalize = False,
#                                               padding = 3,
#                                               range = None,
#                                               scale_each = True,
#                                               pad_value = 1.0)[0]
    
#     tiled_pred  = torchvision.utils.make_grid(torch_pred, 
#                                           nrow = 3, # 순서대로 옆으로 5장씩 0,1,2,3,5 다음줄 6,7,8,9,10 이런식
#                                           normalize = False,
#                                           padding = 3,
#                                           range = None,
#                                           scale_each = True,
#                                           pad_value = 1.0)[0]
    
# #     print(tiled_pred.shape) torch.Size([27249, 1602])
#     tiled_mask = change_value_for_visualize(tiled_mask)
#     tiled_pred = change_value_for_visualize(tiled_pred)
    
#     plt.figure(figsize=figsize) # figsize = (가로,세로)
#     plt.subplot(1,3,1)
#     plt.imshow(tiled_img.numpy(), 'gray')
    
#     plt.subplot(1,3,2)
#     plt.imshow(tiled_img.numpy(), 'gray')
#     plt.imshow(tiled_mask.numpy(), alpha=0.5)
    
#     plt.subplot(1,3,3)
#     plt.imshow(tiled_img.numpy(), 'gray')
#     plt.imshow(tiled_pred.numpy(), alpha=0.5)
    
#     if save is not None:
#         plt.savefig(save)
#     else : 
#         plt.show()
    
#     plt.cla() 
#     plt.clf()
#     plt.close('all')