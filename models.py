import torch 
import torch.nn as nn

# UpTask
from arch.sunggu_inception_v3 import inception_v3
from arch.sunggu_resnet import resnet50

# DownTask
from arch.general_fracture_models import Fracture_Model, Fracture_Model_PedXNet
from arch.rsna_baa_models import BAA_Model, BAA_Model_PedXNet



# Create Model
def get_model(name):
    
    # UpTask
    if name == 'Uptask_PedXNet_7Class_ResNet50':
        model = resnet50(pretrained=False, num_classes=7)
    
    elif name == 'Uptask_PedXNet_7Class_InceptionV3':
        model = inception_v3(pretrained=False, aux_logits=True, num_classes=7)
    
    # DownTask - General Fracture
    elif name == 'Downtask_General_Fracture':
        model = Fracture_Model(pretrained=False, input_channel=1, num_classes=7)

    elif name == 'Downtask_General_Fracture_ImageNet':
        model = Fracture_Model(pretrained=True, input_channel=3, num_classes=1000)

    elif name == 'Downtask_General_Fracture_PedXNet_7Class':
        model = Fracture_Model_PedXNet(pedxnet_class=7)

    elif name == 'Downtask_General_Fracture_PedXNet_30Class':
        model = Fracture_Model_PedXNet(pedxnet_class=30)

    elif name == 'Downtask_General_Fracture_PedXNet_68Class':
        model = Fracture_Model_PedXNet(pedxnet_class=68)                
    
    # DownTask - RSNA Boneage
    elif name == 'Downtask_RSNA_Boneage':
        model = BAA_Model(pretrained=False, input_channel=1, num_classes=7)

    elif name == 'Downtask_RSNA_Boneage_ImageNet':
        model = BAA_Model(pretrained=True, input_channel=3, num_classes=1000)

    elif name == 'Downtask_RSNA_Boneage_PedXNet_7Class':
        model = BAA_Model_PedXNet(pedxnet_class=7)

    elif name == 'Downtask_RSNA_Boneage_PedXNet_30Class':
        model = BAA_Model_PedXNet(pedxnet_class=30)

    elif name == 'Downtask_RSNA_Boneage_PedXNet_68Class':
        model = BAA_Model_PedXNet(pedxnet_class=68)
    
    else :
        raise KeyError("Wrong model name `{}`".format(name))
        
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of Learnable Params:', n_parameters)   

    return model
