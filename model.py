import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch)
        )
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        return F.relu(self.conv(x) + self.shortcut(x))

class SiameseResUNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Two-stream Encoder
        self.eo_enc = ResidualBlock(3, 64)
        self.sar_enc = ResidualBlock(1, 64)

        self.pool = nn.MaxPool2d(2) 

        self.bottleneck = ResidualBlock(128, 256)

        # Decoder with Skip Connections
        self.up1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec1 = ResidualBlock(256, 128)

        self.final = nn.Conv2d(128, 1, 1)

    def forward(self, eo, sar):
        e1 = self.eo_enc(eo)
        s1 = self.sar_enc(sar)
        skip = torch.cat([e1, s1], dim=1) 

        # Downsample
        b = self.pool(skip)
        b = self.bottleneck(b)

        # Upsample + Concatenate Skip
        u1 = self.up1(b)
        u1 = torch.cat([u1, skip], dim=1)
        u1 = self.dec1(u1)

        return torch.sigmoid(self.final(u1))
