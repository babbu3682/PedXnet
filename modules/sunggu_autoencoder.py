'''
@Description: 
@Author: GuoYi
@Date: 2020-06-15 10:04:32
@LastEditTime: 2020-06-28 17:43:02
@LastEditors: GuoYi
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


# util
##***********************************************************************************************************
class AE_Down(nn.Module):
    """
    input:N*C*H*W
    """
    def __init__(self, in_channels, out_channels):
        super(AE_Down, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class AE_Up(nn.Module):
    """
    input:N*C*H*W
    """
    def __init__(self, in_channels, out_channels):
        super(AE_Up, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)
        

##******************************************************************************************************************************
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.lay1     = AE_Down(in_channels=1, out_channels=64)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.lay2     = AE_Down(in_channels=64, out_channels=128)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.lay3     = AE_Down(in_channels=128, out_channels=256)
        self.lay4     = AE_Down(in_channels=256, out_channels=256)

        self.lay5     = AE_Up(in_channels=256, out_channels=256)
        self.lay6     = AE_Up(in_channels=256, out_channels=128)
        self.deconv1  = nn.ConvTranspose2d(128, 128, kernel_size=(2,2), stride=(2,2))
        self.lay7     = AE_Up(in_channels=128, out_channels=64)
        self.deconv2  = nn.ConvTranspose2d(64, 64, kernel_size=(2,2), stride=(2,2))
        self.lay8     = AE_Up(in_channels=64, out_channels=32)
        
        self.head     = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=(1, 1))

    def feature_extract(self, x):
        self.eval()
        with torch.no_grad():
            x = self.lay1(x)
            x = self.maxpool1(x)
            x = self.lay2(x)
            x = self.maxpool2(x)
            x = self.lay3(x)
            x = self.lay4(x)

        return x


    def forward(self, x):
        x = self.lay1(x)
        x = self.maxpool1(x)
        x = self.lay2(x)
        x = self.maxpool2(x)
        x = self.lay3(x)
        x = self.lay4(x)

        x = self.lay5(x)
        x = self.lay6(x)
        x = self.deconv1(x)
        x = self.lay7(x)
        x = self.deconv2(x)
        x = self.lay8(x)

        out = self.head(x)
        out = F.relu(out)
        
        return out


##******************************************************************************************************************************
from torch import Tensor
from typing import Type, Any, Callable, Union, List, Optional

model_urls = {
    # imagenet pretrained
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 7,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups     = groups
        self.base_width = width_per_group
        self.conv1      = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1        = norm_layer(self.inplanes)
        self.relu       = nn.ReLU(inplace=True)
        self.maxpool    = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1     = self._make_layer(block, 64, layers[0])
        self.layer2     = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3     = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4     = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        
        # for downstream
        # self.avgpool    = nn.AdaptiveAvgPool2d((1, 1))     
        # self.fc         = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int, stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride), norm_layer(planes * block.expansion),)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # print("0 == ", x.shape)   torch.Size([16, 1, 512, 512])  (input: 512x512 기준)
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))  # stem
        # print("1 == ", x.shape)   torch.Size([16, 64, 256, 256])  
        x = self.layer1(x)        # stage1
        # print("2 == ", x.shape)   torch.Size([16, 256, 128, 128])
        x = self.layer2(x)                      # stage2
        # print("3 == ", x.shape)   torch.Size([16, 512, 64, 64])
        x = self.layer3(x)                      # stage3
        # print("4 == ", x.shape)   torch.Size([16, 1024, 32, 32])
        x = self.layer4(x)                      # stage4
        # print("5 == ", x.shape)   torch.Size([16, 2048, 16, 16])

        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


class PS_Decoder_Block(nn.Module):
    def __init__(self, in_channels, out_channels, up_scale=2):
        super(PS_Decoder_Block, self).__init__()

        self.up = nn.Sequential(
            nn.Conv2d(in_channels, in_channels*(up_scale**2), kernel_size=3, padding=1), 
            nn.PixelShuffle(2), 
            nn.ReLU(True),
            )

        self.two_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            )

    def forward(self, x):
        x = self.up(x)
        x = self.two_conv(x)
        return x


class Resnet_AutoEncoder(nn.Module):
    def __init__(self):
        super(Resnet_AutoEncoder, self).__init__()

        self.encoder  = ResNet(block=Bottleneck, layers=[3, 4, 6, 3]) # fix Resnet50 spec

        self.decoder4 = PS_Decoder_Block(in_channels=2048, out_channels=1024)
        self.decoder3 = PS_Decoder_Block(in_channels=1024, out_channels=512)
        self.decoder2 = PS_Decoder_Block(in_channels=512, out_channels=256)
        self.decoder1 = PS_Decoder_Block(in_channels=256, out_channels=64)
        self.decoder0 = PS_Decoder_Block(in_channels=64, out_channels=1)
        
        self.head     = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1)

    def feature_extract(self, x):
        self.eval()
        with torch.no_grad():
            x = self.encoder(x)[0]
        return x

    def forward(self, x):
        x   = self.encoder(x)
        # print("check == ", x.shape) input 512x512 기준, torch.Size([16, 2048, 16, 16])
        x   = self.decoder4(x)
        x   = self.decoder3(x)
        x   = self.decoder2(x)
        x   = self.decoder1(x)
        x   = self.decoder0(x)
        x   = self.head(x)
        x   = F.relu(x)
        return x