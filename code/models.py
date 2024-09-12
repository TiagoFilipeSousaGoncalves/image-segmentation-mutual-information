# Imports
from collections import OrderedDict

# PyTorch Imports
import torch
import torch.nn as nn
import torchvision
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models.segmentation.fcn import FCNHead
import torch.nn.functional as F



# Class: FCN
class FCN(nn.Module):

    def __init__(self, out_channels=21):
        super(FCN, self).__init__()
        self.fcn = torchvision.models.segmentation.fcn_resnet50(pretrained=True, progress=True)
        self.fcn.classifier = FCNHead(2048, out_channels)

        return


    def forward(self, x):
        
        return self.fcn(x)



# Class: DeepLabV3, from https://towardsdatascience.com/transfer-learning-for-segmentation-using-deeplabv3-in-pytorch-f770863d6a42
class DeepLabV3(nn.Module):

    def __init__(self, out_channels=21):
        super(DeepLabV3, self).__init__()
        self.deeplabv3 = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True, progress=True)
        self.deeplabv3.classifier = DeepLabHead(2048, out_channels)

        return


    def forward(self, x):

        out = self.deeplabv3(x)['out']
        
        return out



# Class: DeepLabV3 w/ ResNest-50
class DeepLabV3_resnest(nn.Module):

    def __init__(self, out_channels=21):
        super(DeepLabV3_resnest, self).__init__()

        backbone = torch.hub.load('zhanghang1989/ResNeSt', 'resnest50', pretrained=True)
        res = list(backbone.children())[:-2]
        res = nn.Sequential(*res)

        self.backbone = res
        self.classifier = DeepLabHead(2048, out_channels)

    def forward(self, x):
        input_shape = x.shape[-2:]

        x = self.backbone(x)
        x = self.classifier(x)

        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)

        result = OrderedDict()
        result["out"] = x

        return result



# Class: U-Net, from https://pytorch.org/hub/mateuszbuda_brain-segmentation-pytorch_unet/ with sigmoid removed
class UNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(UNet, self).__init__()

        features = init_features
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        
        return self.conv(dec1)


    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )