import torch
import torch.nn as nn

class RecurrentBlock3D(nn.Module):
    def __init__(self, ch, t=2, norm="INSTANCE"):
        super().__init__()
        Norm = nn.InstanceNorm3d if norm.upper().startswith("INSTANCE") else nn.BatchNorm3d
        self.t = t
        self.block = nn.Sequential(
            nn.Conv3d(ch, ch, 3, padding=1, bias=False),
            Norm(ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x1=None
        for i in range(self.t):
            x1 = self.block(x if i==0 else (x+x1))
        return x1

class RRCNNBlock3D(nn.Module):
    def __init__(self, ch_in, ch_out, t=2, norm="INSTANCE"):
        super().__init__()
        Norm = nn.InstanceNorm3d if norm.upper().startswith("INSTANCE") else nn.BatchNorm3d
        self.conv_1x1 = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, 1, bias=False),
            Norm(ch_out),
            nn.ReLU(inplace=True),
        )
        self.rcnn = nn.Sequential(
            RecurrentBlock3D(ch_out, t=t, norm=norm),
            RecurrentBlock3D(ch_out, t=t, norm=norm),
        )
    def forward(self, x):
        x = self.conv_1x1(x)
        out = self.rcnn(x)
        return x + out

class R2UNet3D(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, channels=(32,64,128,256,320), t=2, norm="INSTANCE"):
        super().__init__()
        c1,c2,c3,c4,c5 = channels
        self.enc1 = RRCNNBlock3D(in_ch, c1, t, norm); self.pool1 = nn.MaxPool3d(2)
        self.enc2 = RRCNNBlock3D(c1, c2, t, norm);    self.pool2 = nn.MaxPool3d(2)
        self.enc3 = RRCNNBlock3D(c2, c3, t, norm);    self.pool3 = nn.MaxPool3d(2)
        self.enc4 = RRCNNBlock3D(c3, c4, t, norm);    self.pool4 = nn.MaxPool3d(2)
        self.bott = RRCNNBlock3D(c4, c5, t, norm)
        self.up4 = nn.ConvTranspose3d(c5, c4, 2, 2); self.dec4 = RRCNNBlock3D(c4+c4, c4, t, norm)
        self.up3 = nn.ConvTranspose3d(c4, c3, 2, 2); self.dec3 = RRCNNBlock3D(c3+c3, c3, t, norm)
        self.up2 = nn.ConvTranspose3d(c3, c2, 2, 2); self.dec2 = RRCNNBlock3D(c2+c2, c2, t, norm)
        self.up1 = nn.ConvTranspose3d(c2, c1, 2, 2); self.dec1 = RRCNNBlock3D(c1+c1, c1, t, norm)
        self.out_conv = nn.Conv3d(c1, out_ch, 1)
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        bt = self.bott(self.pool4(e4))
        d4 = self.dec4(torch.cat([self.up4(bt), e4], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.out_conv(d1)
