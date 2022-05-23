import torch 
import torch.nn as nn

# UpTask
    # Supervised Model  
# from arch.sunggu_resnet import resnet50
from arch.sunggu_autoencoder import Resnet_AutoEncoder
from arch.general_fracture_models import General_Fracture_Model
from arch.rsna_baa_models import *
from arch.pneumonia_models import *



    # Unsupervised Model - Previous Works 
# from arch.sunggu_autoencoder import Resnet_AutoEncoder      # Model Genesis
# from arch.sunggu_modelgenensis import Resnet_AutoEncoder    # Trans VW



# DownTask
    # General Fracture



def Downtask_Model(task_name, pretrained, freeze_backbone, backbone_name):
    if task_name == '1.General_Fracture':
        model = General_Fracture_Model(backbone_name=backbone_name)
        
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


# Create Model
def create_model(stream, name):
    if stream == 'Upstream':
        # Ours
        if name == 'Uptask_Sup_Classifier':
            # model = Supervised_Classifier_Model(pretrained)
            pass
        
        # Previous works
        elif name == 'Uptask_Unsup_AutoEncoder':
            model = Resnet_AutoEncoder()

        # elif name == 'Unsupervised_ModelGenesis_Model':
        #     model = ModelGenesis()

        # elif name == 'Unsupervised_TransVW_Model':
            # model = TransVW()            


    elif stream == 'Downstream':
        # Need for Customizing ... !
        if name == 'Downtask_General_Fracture':
            model = General_Fracture_Model()
        
        elif name == 'Downtask_RSNA_Boneage':
            model = RSNA_BAA_Model()
        
        elif name == 'Downtask_Pneumonia':
            model = Pneumonia_Model()
        


    else :
        raise KeyError("Wrong model name `{}`".format(name))
        
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of Learnable Params:', n_parameters)   

    return model


