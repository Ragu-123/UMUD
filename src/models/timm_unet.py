import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False

from .unet import DoubleConv, UpBlock, StandaloneUNet


class TimmUNet(nn.Module):
    """
    Modular U-Net with timm pre-trained encoders (ResNet, EfficientNet, etc.).
    """
    def __init__(self, backbone='resnet34', in_channels=3, num_classes=1, pretrained=True):
        super().__init__()
        if not HAS_TIMM:
            raise ImportError("timm is not installed. Install via pip install timm.")

        self.encoder = timm.create_model(
            backbone,
            pretrained=pretrained,
            in_chans=in_channels,
            features_only=True
        )
        encoder_channels = self.encoder.feature_info.channels()

        self.dec4 = UpBlock(encoder_channels[4], encoder_channels[3], 256)
        self.dec3 = UpBlock(256, encoder_channels[2], 128)
        self.dec2 = UpBlock(128, encoder_channels[1], 64)
        self.dec1 = UpBlock(64, encoder_channels[0], 32)

        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            DoubleConv(16, 16),
            nn.Conv2d(16, num_classes, kernel_size=1)
        )

    def forward(self, x):
        h, w = x.shape[2:]
        feats = self.encoder(x)

        d4 = self.dec4(feats[4], feats[3])
        d3 = self.dec3(d4, feats[2])
        d2 = self.dec2(d3, feats[1])
        d1 = self.dec1(d2, feats[0])
        out = self.final_up(d1)

        if out.shape[2:] != (h, w):
            out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=False)
        return out


def build_segmentation_model(backbone='resnet34', in_channels=3, num_classes=1, pretrained=True):
    """
    Factory function returning a pre-trained TimmUNet or StandaloneUNet fallback.
    """
    if HAS_TIMM and backbone != 'standalone':
        try:
            return TimmUNet(backbone=backbone, in_channels=in_channels, num_classes=num_classes, pretrained=pretrained)
        except Exception:
            return StandaloneUNet(in_channels=in_channels, num_classes=num_classes)
    return StandaloneUNet(in_channels=in_channels, num_classes=num_classes)
