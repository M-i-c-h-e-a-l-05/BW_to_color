"""
Colorization model: U-Net that predicts Lab 'a,b' color channels from the
'L' (lightness/grayscale) channel.

Input:  (batch, 1, H, W)   -- the L channel, normalized to [-1, 1]
Output: (batch, 2, H, W)   -- predicted a,b channels, in [-1, 1]
"""
import torch
import torch.nn as nn


def conv_block(in_ch, out_ch):
    """Two 3x3 convs + BatchNorm + ReLU. The basic building block of the U-Net."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class UNetColorizer(nn.Module):
    def __init__(self, base_channels: int = 64):
        super().__init__()
        c = base_channels

        # Encoder (downsampling path)
        self.enc1 = conv_block(1, c)          # L channel in
        self.enc2 = conv_block(c, c * 2)
        self.enc3 = conv_block(c * 2, c * 4)
        self.enc4 = conv_block(c * 4, c * 8)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = conv_block(c * 8, c * 16)

        # Decoder (upsampling path) with skip connections from the encoder
        self.up4 = nn.ConvTranspose2d(c * 16, c * 8, 2, stride=2)
        self.dec4 = conv_block(c * 16, c * 8)   # concatenated with enc4 skip

        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, 2, stride=2)
        self.dec3 = conv_block(c * 8, c * 4)    # concatenated with enc3 skip

        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)
        self.dec2 = conv_block(c * 4, c * 2)    # concatenated with enc2 skip

        self.up1 = nn.ConvTranspose2d(c * 2, c, 2, stride=2)
        self.dec1 = conv_block(c * 2, c)        # concatenated with enc1 skip

        # Final 1x1 conv -> 2 channels (a, b), tanh squashes to [-1, 1]
        self.out_conv = nn.Conv2d(c, 2, kernel_size=1)
        self.out_act = nn.Tanh()

    def forward(self, x):
        # x: (B, 1, H, W)  -- the L channel

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        out = self.out_conv(d1)
        return self.out_act(out)  # (B, 2, H, W), values in [-1, 1]


if __name__ == "__main__":
    # Quick smoke test: does a forward pass run and produce the right shape?
    model = UNetColorizer(base_channels=64)
    dummy_L = torch.randn(2, 1, 256, 256)
    out = model(dummy_L)
    print("Input shape :", dummy_L.shape)
    print("Output shape:", out.shape)  # expect (2, 2, 256, 256)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")
