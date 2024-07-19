import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import logging

logger = logging.getLogger('xairesunet_logger')

class XAIResUNet(nn.Module):
    def __init__(self, num_classes=1, pretrained=True, weights_path=None, verbose=False, dropout_rate=0.5):
        super(XAIResUNet, self).__init__()
        self.verbose = verbose

        if weights_path:
            resnet = models.resnet50(weights=None)         
        elif pretrained:
            logger.info("Loading Imagenet Weights..")
            resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        else:
            logger.info("No weights path provided or pretrained specified.. initializing randomly")
            resnet = models.resnet50(weights=None)

        # Use the first parts of the resnet directly as encoder blocks
        self.enc1 = nn.Sequential(*list(resnet.children())[:3])  # Initial conv, bn, and relu
        self.enc2 = nn.Sequential(*list(resnet.children())[3:5])  # MaxPool and layer1
        self.enc3 = resnet.layer2  # Layer2
        self.enc4 = resnet.layer3  # Layer3
        self.enc5 = resnet.layer4  # Layer4

        self.encoder = nn.Sequential(self.enc1, self.enc2, self.enc3, self.enc4, self.enc5)

        if weights_path:
            logger.info("Loading RadImageNet Weights..")
            self.load_pretrained_weights(weights_path)

        # Bottleneck
        self.bottleneck = self.double_conv(2048, 2048)

        # Decoder blocks with skip connections
        self.up5 = self.decoder_block(2048, 1024, dropout_rate)
        self.up4 = self.decoder_block(1024 + 1024, 512, dropout_rate)
        self.up3 = self.decoder_block(512 + 512, 256, dropout_rate)
        self.up2 = self.decoder_block(256 + 256, 64, dropout_rate)
        self.up1 = self.decoder_block(64 + 64, 64, dropout_rate)

        # Final output
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )

    def decoder_block(self, in_channels, out_channels, dropout_rate):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout_rate),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        # Encoder path
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)

        # Bottleneck
        bottleneck = self.bottleneck(enc5)

        # Decoder path
        dec5 = self.up5(bottleneck)
        dec4 = self.up4(torch.cat([dec5, enc4], dim=1))
        dec3 = self.up3(torch.cat([dec4, enc3], dim=1))
        dec2 = self.up2(torch.cat([dec3, enc2], dim=1))
        dec1 = self.up1(torch.cat([dec2, enc1], dim=1))


        output = self.final_conv(dec1)
    
        return output

    def load_pretrained_weights(self, weights_path):
        state_dict = torch.load(weights_path, map_location='cpu')
        encoder_dict = self.encoder.state_dict()
        # Filter out unnecessary keys
        pretrained_dict = {k: v for k, v in state_dict.items() if k in encoder_dict and encoder_dict[k].size() == v.size()}
        encoder_dict.update(pretrained_dict)
        self.encoder.load_state_dict(encoder_dict)
        if self.verbose:
            logger.info("Loaded pretrained weights for the encoder.")

