import torch.nn as nn
import mynn

__all__ = ['Segmodel']


class Segmodel(nn.Module):
    def __init__(self, backbone, fchannel=0, classes=2, dropout=0.5, zoom_factor=None):
        super(Segmodel, self).__init__()
        assert classes > 1
        self.backbone = backbone
        outchannel = self.backbone.outchannel
        self.flayer = None
        if fchannel > 0:
            self.flayer = nn.Sequential(
                nn.Conv2d(outchannel, fchannel, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(fchannel),
                nn.ReLU(inplace=True)
            )
            outchannel = fchannel
        self.dropout = nn.Dropout2d(p=dropout, inplace=False)
        self.upsample = None
        if zoom_factor is not None:
            self.upsample = mynn.mymodules.InterpBilinear2d(zoom_factor=zoom_factor)
        self.classify = nn.Conv2d(outchannel, classes, kernel_size=1, bias=True)

    def forward(self, x):
        x = self.backbone(x)
        if self.flayer is not None:
            x = self.flayer(x)
        x = self.dropout(x)
        if self.upsample is not None:
            x = self.upsample(x)
        x = self.classify(x)
        return x
