import torch
import torch.nn as nn
from torch import Tensor

from arch.sunggu_resnet import ResNet, Bottleneck
from arch.sunggu_inception_v3 import inception_v3



class ResNet_Feature_Extractor(ResNet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        del self.fc
        del self.avgpool

    def forward(self, x: Tensor) -> Tensor:
        # Stem
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))

        # Four Stage
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def load_state_dict(self, state_dict, **kwargs):
        # state_dict.pop("fc.bias")
        # state_dict.pop("fc.weight")
        super().load_state_dict(state_dict, **kwargs)



# class RSNA_BAA_Model(nn.Module):
#     def __init__(self):
#         super(RSNA_BAA_Model, self).__init__()
        

#         self.encoder = ResNet_Feature_Extractor(block=Bottleneck, layers=[3, 4, 6, 3])
#         self.pool  = nn.AdaptiveAvgPool2d(1)
        
#         self.fc1   = nn.Linear(2048+1, 1024)  # +1 is for gender.
#         self.relu1 = nn.ReLU()
#         self.drop1 = nn.Dropout(0.5)
        
#         self.fc2   = nn.Linear(1024, 1024)
#         self.relu2 = nn.ReLU()
#         self.drop2 = nn.Dropout(0.5)
        
#         self.head   = nn.Linear(1024, 1)
        
#     def forward(self, x, gender):
#         x = self.encoder(x)
#         x = self.pool(x)
#         x = x.view(x.shape[0], -1)
        
#         x = torch.cat([x, gender], dim=1)
        
#         x = self.drop1(self.relu1(self.fc1(x)))
#         x = self.drop2(self.relu2(self.fc2(x)))
#         x = self.head(x)
        
#         return x



class BAA_Model(nn.Module):
    def __init__(self, pretrained=False, input_channel=3, num_classes=1000):
        super(BAA_Model, self).__init__()
        self.input_channel = input_channel
        self.feat_extractor = nn.Sequential(*list(inception_v3(pretrained=pretrained, aux_logits=False, init_weights=True, input_channel=input_channel, num_classes=num_classes).children())[:-3])
        self.pool  = nn.AdaptiveAvgPool2d(1)
        
        self.fc1   = nn.Linear(2048+1, 1024) # +1 is for gender.
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.5)
        
        self.fc2   = nn.Linear(1024, 1024)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(0.5)
        
        self.head   = nn.Linear(1024, 1)
        
    def forward(self, x, gender):
        if self.input_channel == 3:
            # Image net require RGB channel
            x = x.repeat(1, 3, 1, 1)

        x = self.feat_extractor(x)
        x = self.pool(x)
        x = x.view(x.shape[0], -1)
        
        x = torch.cat([x, gender], dim=1)

        x = self.drop1(self.relu1(self.fc1(x)))
        x = self.drop2(self.relu2(self.fc2(x)))
        x = self.head(x)
        
        return x



class BAA_Model_PedXNet(nn.Module):
    def __init__(self, pedxnet_class=7):
        super(BAA_Model_PedXNet, self).__init__()

        if pedxnet_class==7:
            inception = inception_v3(pretrained=False, aux_logits=False, init_weights=False, input_channel=1, num_classes=7)
            print("Before Weight ==> ", inception.state_dict()['Conv2d_1a_3x3.conv.weight'].mean().item())
            state_dict = torch.load('/workspace/sunggu/3.Child/PedXnet_Code_Factory/checkpoints/PedXnet-InceptionV3-7class/epoch_40_best_metric_model.pth')['model_state_dict']

        elif pedxnet_class==30:
            inception = inception_v3(pretrained=False, aux_logits=False, init_weights=False, input_channel=1, num_classes=30)
            print("Before Weight ==> ", inception.state_dict()['Conv2d_1a_3x3.conv.weight'].mean().item())
            state_dict = torch.load('/workspace/sunggu/3.Child/PedXnet_Code_Factory/checkpoints/PedXnet-InceptionV3-30class/epoch_50_best_metric_model.pth')['model_state_dict']

        elif pedxnet_class==68:
            inception = inception_v3(pretrained=False, aux_logits=False, init_weights=False, input_channel=1, num_classes=68)
            print("Before Weight ==> ", inception.state_dict()['Conv2d_1a_3x3.conv.weight'].mean().item())
            state_dict = torch.load('/workspace/sunggu/3.Child/PedXnet_Code_Factory/checkpoints/PedXnet-InceptionV3-68class/epoch_80_best_metric_model.pth')['model_state_dict']

        unexpected_keys = ['AuxLogits.conv0.conv.weight', 'AuxLogits.conv0.bn.weight', 'AuxLogits.conv0.bn.bias', 'AuxLogits.conv0.bn.running_mean', 'AuxLogits.conv0.bn.running_var', 'AuxLogits.conv0.bn.num_batches_tracked', 'AuxLogits.conv1.conv.weight', 'AuxLogits.conv1.bn.weight', 'AuxLogits.conv1.bn.bias', 'AuxLogits.conv1.bn.running_mean', 'AuxLogits.conv1.bn.running_var', 'AuxLogits.conv1.bn.num_batches_tracked', 'AuxLogits.fc.weight', 'AuxLogits.fc.bias']
        for key in unexpected_keys:
            if key in state_dict:
                del state_dict[key]
        inception.load_state_dict(state_dict, strict=False)
        print("After Weight ==> ", inception.state_dict()['Conv2d_1a_3x3.conv.weight'].mean().item())

        self.feat_extractor = nn.Sequential(*list(inception.children())[:-3])
        self.pool  = nn.AdaptiveAvgPool2d(1)
        
        self.fc1   = nn.Linear(2048+1, 1024) # +1 is for gender.
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
