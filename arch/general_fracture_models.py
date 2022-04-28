import torch
import torch.nn as nn
from torch import Tensor
from sunggu_resnet import ResNet, Bottleneck


class ResNet_Feature_Extractor(ResNet(block=Bottleneck, layers=[3, 4, 6, 3])):
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




# DownTask
    # General Fracture
class General_Fracture_Model(nn.Module):
    def __init__(self):
        super(General_Fracture_Model, self).__init__()

        self.feat_extractor = ResNet_Feature_Extractor()

        self.pool  = nn.AdaptiveAvgPool2d(1)
        
        self.fc1   = nn.Linear(2048, 1024)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.5)
        
        self.fc2   = nn.Linear(1024, 1024)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(0.5)
        
        self.head  = nn.Linear(1024, 1)
        
    def forward(self, x):
        x = self.feat_extractor(x)
        x = self.pool(x)
        x = x.view(x.shape[0], -1)
        
        x = self.drop1(self.relu1(self.fc1(x)))
        x = self.drop2(self.relu2(self.fc2(x)))
        x = self.head(x)
        
        return x

