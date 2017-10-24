import torch.nn as nn
import mynn

__all__ = ['ReSegResmodel']


class ReSegResmodel(nn.Module):
    def __init__(self, backbone, fchannel=128, hiddenchannel=128, renet_num=2, classes=2, dropout=0.5, zoom_factor=None):
        super(ReSegResmodel, self).__init__()
        assert classes > 1
        assert renet_num > 0
        self.backbone = backbone
        outchannel = self.backbone.outchannel
        self.flayer = nn.Sequential(
            nn.Conv2d(outchannel, fchannel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fchannel),
            nn.ReLU(inplace=True)
        )
        outchannel = fchannel
        renetlayers = [mynn.ReNetLayer(outchannel, hiddenchannel, direction="height"),
                       mynn.ReNetLayer(2*hiddenchannel, hiddenchannel, direction="width")]
        outchannel = 2*hiddenchannel
        for _ in range(renet_num - 1):
            renetlayers.append(mynn.ReNetLayer(outchannel, hiddenchannel, direction="height"))
            renetlayers.append(mynn.ReNetLayer(2*hiddenchannel, hiddenchannel, direction="height"))
            outchannel = 2 * hiddenchannel
        self.renet = nn.Sequential(*renetlayers)
        self.flayer2 = nn.Sequential(
            nn.Conv2d(outchannel, fchannel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fchannel),
        )
        outchannel = fchannel
        self.dropout = nn.Dropout2d(p=dropout, inplace=False)
        self.upsample = None
        if zoom_factor is not None:
            self.upsample = mynn.mymodules.InterpBilinear2d(zoom_factor=zoom_factor)
        self.classify = nn.Conv2d(outchannel, classes, kernel_size=1, bias=True)

    def forward(self, x):
        x = self.backbone(x)
        x = self.flayer(x)
        res = self.renet(x)
        res = self.flayer2(res)
        x = x + res
        x = nn.ReLU(inplace=True)(x)
        x = self.dropout(x)
        if self.upsample is not None:
            x = self.upsample(x)
        x = self.classify(x)
        return x
