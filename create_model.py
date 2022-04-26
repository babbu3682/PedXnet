import torch 
import torch.nn as nn
######################################################                    UpTask                          ########################################################

###################################################### Supervised Model  
from modules.sunggu_resnet import resnet50, resnet152
from modules.sunggu_inception_v3 import inception_v3
from modules.sunggu_autoencoder import Resnet_AutoEncoder
from modules.sunggu_modelgenensis import Resnet_AutoEncoder



# def Supervised_Classifier_Model(pretrained):
#     if backbone_name == 'ResNet':
#         model        = resnet50(pretrained=False, num_classes=68)
#     elif backbone_name == 'InceptionV3':
#         model        = inception_v3(pretrained=False, aux_logits=True, num_classes=68)
#     else :
#         raise Exception('Error...! backbone_name')  

#     if pretrained is not None:
#         if backbone_name == 'ResNet':
#             check_point    = torch.load(pretrained)
#             model_dict     = model.state_dict()
#             print("Check Before weight = ", model_dict['encoder.conv1.weight'][0])
#             pretrained_dict = {k: v for k, v in check_point['model_state_dict'].items() if k in model_dict and 'encoder.' in k}
#             model_dict.update(pretrained_dict) 
#             model.load_state_dict(model_dict)        
#             print("Check After weight = ", model_dict['encoder.conv1.weight'][0])
#             print("Succeed Load Pretrained...!")   
#         elif backbone_name == 'InceptionV3' :
#             check_point     = torch.load(pretrained)
#             model_dict      = model.state_dict()
#             print("Check Before weight = ", model_dict['encoder._conv_stem.weight'][0])
#             pretrained_dict = {k: v for k, v in check_point['model_state_dict'].items() if k in model_dict and 'encoder.' in k}
#             model_dict.update(pretrained_dict) 
#             model.load_state_dict(model_dict)        
#             print("Check After weight = ", model_dict['encoder._conv_stem.weight'][0])
#             print("Succeed Load Pretrained...!")   
#         else :
#             raise Exception('Error...! backbone_name')  

#     n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print('Number of Learnable Params:', n_parameters)
#     return model



###################################################### Unsupervised Model - Previous Works 
# Model Genesis
# Trans VW



######################################################                    DownTask                          ########################################################

# 개선 필요
'''
# 1. General Fracture
class Bone_model(nn.Module):
    def __init__(self, freeze_backbone=freeze_backbone, backbone_name=backbone_name):
        super(Bone_model, self).__init__()
        self.freeze_backbone = freeze_backbone

        if backbone_name == 'resnet50':
            self.feat_extractor = nn.Sequential(*list(resnet50(pretrained=False, aux_logits=False, init_weights=True).children())[:-3])
        elif backbone_name == 'inceptionV3':
            self.feat_extractor = nn.Sequential(*list(inception_v3(pretrained=False, aux_logits=False, init_weights=True).children())[:-3])            
        else :
            raise Exception('Error...! backbone_name')    

        self.pool  = nn.AdaptiveAvgPool2d(1)
        
        self.fc1   = nn.Linear(2049, 1024)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.5)
        
        self.fc2   = nn.Linear(1024, 1024)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(0.5)
        
        self.head   = nn.Linear(1024, 1)
        
    def forward(self, x, gender):
        if self.freeze_backbone:
            with torch.no_grad():
                x = self.feat_extractor(x)
        else :
            x = self.feat_extractor(x)

        x = self.pool(x)
        x = x.view(x.shape[0], -1)
        
        x = torch.cat([x, gender], dim=1)
        
        x = self.drop1(self.relu1(self.fc1(x)))
        x = self.drop2(self.relu2(self.fc2(x)))
        x = self.head(x)
        
        return x

# 2. RSNA BoneAge
class RSNA_BoneAge_Model(nn.Module):
    def __init__(self):
        super(RSNA_BoneAge_Model, self).__init__()
        
        self.feat_extractor = nn.Sequential(*list(inception_v3(pretrained=False, aux_logits=False, init_weights=True).children())[:-3])
        self.pool  = nn.AdaptiveAvgPool2d(1)
        
        self.fc1   = nn.Linear(2049, 1024)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.5)
        
        self.fc2   = nn.Linear(1024, 1024)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(0.5)
        
        self.head   = nn.Linear(1024, 1)
        
    def forward(self, x, gender):
        x = self.feat_extractor(x)
        x = self.pool(x)
        x = x.view(x.shape[0], -1)
        
        x = torch.cat([x, gender], dim=1)
        
        x = self.drop1(self.relu1(self.fc1(x)))
        x = self.drop2(self.relu2(self.fc2(x)))
        x = self.head(x)
        
        return x
    
# 3. Nodule Detection
class Bone_model(nn.Module):
    def __init__(self):
        super(Bone_model, self).__init__()
        
        self.feat_extractor = nn.Sequential(*list(inception_v3(pretrained=False, aux_logits=False, init_weights=True).children())[:-3])
        self.pool  = nn.AdaptiveAvgPool2d(1)
        
        self.fc1   = nn.Linear(2049, 1024)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.5)
        
        self.fc2   = nn.Linear(1024, 1024)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(0.5)
        
        self.head   = nn.Linear(1024, 1)
        
    def forward(self, x, gender):
        x = self.feat_extractor(x)
        x = self.pool(x)
        x = x.view(x.shape[0], -1)
        
        x = torch.cat([x, gender], dim=1)
        
        x = self.drop1(self.relu1(self.fc1(x)))
        x = self.drop2(self.relu2(self.fc2(x)))
        x = self.head(x)
        
        return x



def Downtask_Model(task_name, pretrained, freeze_backbone, backbone_name):
    if task_name == '1.General_Fracture':
        model = Patient_Cls_model(end2end=end2end, backbone_name=backbone_name)
    elif task_name == '2.RSNA_BoneAge':
        model = Patient_Cls_model(end2end=end2end, backbone_name=backbone_name)
    elif task_name == '3.Nodule_Detection':
        model = Patient_Cls_model(end2end=end2end, backbone_name=backbone_name)        
    else : 
        raise Exception('Error...! task_name')    

    if pretrained is not None:
        if backbone_name == 'resnet50':
            check_point     = torch.load(pretrained)
            model_dict      = model.state_dict()
            print("Check Before weight = ", model_dict['encoder.conv1.weight'][0])
            pretrained_dict = {k: v for k, v in check_point['model_state_dict'].items() if k in model_dict and 'encoder.' in k}
            model_dict.update(pretrained_dict) 
            model.load_state_dict(model_dict)        
            print("Check After weight = ", model_dict['encoder.conv1.weight'][0])
            print("Succeed Load Pretrained...!")   

        elif backbone_name == 'resnet50':
            check_point     = torch.load(pretrained)
            model_dict      = model.state_dict()
            print("Check Before weight = ", model_dict['encoder._conv_stem.weight'][0])
            pretrained_dict = {k: v for k, v in check_point['model_state_dict'].items() if k in model_dict and 'encoder.' in k}
            model_dict.update(pretrained_dict) 
            model.load_state_dict(model_dict)        
            print("Check After weight = ", model_dict['encoder._conv_stem.weight'][0])
            print("Succeed Load Pretrained...!")   

        # Previous 까지 고려하기
        else :
            raise Exception('Error...! task_name')  

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of Learnable Params:', n_parameters)

    return model
'''

######################################################################################################################################################################
######################################################                    Create Model                        ########################################################
######################################################################################################################################################################


def create_model(stream, name, pretrained):
    if stream == 'Upstream':
        # Ours
        if name == 'Uptask_Sup_Classifier':
            # model = Supervised_Classifier_Model(pretrained)
            pass
        

        # Previous works
        elif name == 'Uptask_Unsup_AutoEncoder':
            model = Resnet_AutoEncoder()

        elif name == 'Unsupervised_ModelGenesis_Model':
            model = ModelGenesis()

        # elif name == 'Unsupervised_TransVW_Model':
            # model = TransVW()            


    elif stream == 'Downstream':
        if name == '1.General Fracture':
            model = General_Fracture_Model(pretrained)
        elif name == '2.RSNA BoneAge':
            model = RSNA_BoneAge_Model(pretrained)
        elif name == '3.Ped_Pneumo':
            model = resnet50(pretrained) 
            path = '/mnt/nas100/sunggu/3.Child/models/[Uptask][Unsup]ResNet_AutoEncoder_DataParallel/epoch_88_checkpoint.pth'
            state_dict = torch.load(path)
            model.load_state_dict(state_dict['model_state_dict'])


    else :
        raise KeyError("Wrong model name `{}`".format(name))
        
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of Learnable Params:', n_parameters)   

    return model


