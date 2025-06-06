#model.py
import torch
from torch import nn

class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Config
        in_channels  = 4
        out_channels = 3
        n_filters    = 32
        activation   = nn.ReLU()

        # Up and downsampling methods
        self.downsample  = nn.MaxPool2d((2, 2), stride=2)
        self.upsample    = nn.UpsamplingBilinear2d(scale_factor=2)

        # Encoder
        self.enc_block_1 = EncoderBlock(in_channels, 1 * n_filters, activation)
        self.enc_block_2 = EncoderBlock(1 * n_filters, 2 * n_filters, activation)
        self.enc_block_3 = EncoderBlock(2 * n_filters, 4 * n_filters, activation)
        self.enc_block_4 = EncoderBlock(4 * n_filters, 8 * n_filters, activation)

        # Bottleneck
        self.bottleneck  = nn.Sequential(
            nn.Conv2d( 8 * n_filters, 16 * n_filters, kernel_size=(3, 3), stride=1, padding=1),
            activation,
            nn.Conv2d(16 * n_filters,  8 * n_filters, kernel_size=(3, 3), stride=1, padding=1),
            activation
        )

        # Decoder
        self.dec_block_4 = DecoderBlock(16 * n_filters, 4 * n_filters, activation)
        self.dec_block_3 = DecoderBlock( 8 * n_filters, 2 * n_filters, activation)
        self.dec_block_2 = DecoderBlock( 4 * n_filters, 1 * n_filters, activation)
        self.dec_block_1 = DecoderBlock( 2 * n_filters, 1 * n_filters, activation)

        # Output projection
        self.output      = nn.Conv2d(1 * n_filters,  out_channels, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        # Encoder
        skip_1 = self.enc_block_1(x)
        x      = self.downsample(skip_1)
        skip_2 = self.enc_block_2(x)
        x      = self.downsample(skip_2)
        skip_3 = self.enc_block_3(x)
        x      = self.downsample(skip_3)
        skip_4 = self.enc_block_4(x)
        x      = self.downsample(skip_4)

        # Bottleneck
        x      = self.bottleneck(x)

        # Decoder
        x      = self.upsample(x)
        x      = torch.cat((x, skip_4), axis=1)
        x      = self.dec_block_4(x)
        x      = self.upsample(x)
        x      = torch.cat((x, skip_3), axis=1)
        x      = self.dec_block_3(x)
        x      = self.upsample(x)
        x      = torch.cat((x, skip_2), axis=1)
        x      = self.dec_block_2(x)
        x      = self.upsample(x)
        x      = torch.cat((x, skip_1), axis=1)
        x      = self.dec_block_1(x)
        x      = self.output(x)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation=nn.ReLU()):
        super().__init__()
        self.encoder_block = nn.Sequential(
            nn.Conv2d(in_channels,  out_channels, kernel_size=(3, 3), stride=1, padding=1),
            activation,
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=1, padding=1),
            activation
        )
    def forward(self, x):
        return self.encoder_block(x)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation=nn.ReLU()):
        super().__init__()
        self.decoder_block = nn.Sequential(
            nn.Conv2d(in_channels,  in_channels // 2, kernel_size=(3, 3), stride=1, padding=1),
            activation,
            nn.Conv2d(in_channels // 2, out_channels, kernel_size=(3, 3), stride=1, padding=1),
            activation
        )
    def forward(self, x):
        return self.decoder_block(x)