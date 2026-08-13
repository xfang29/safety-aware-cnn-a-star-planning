"""Neural-network models for learned path preference prediction."""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """
    Two consecutive 3x3 convolution blocks.

    Each convolution is followed by batch normalization and ReLU.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(
                out_channels
            ),
            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(
                out_channels
            ),
            nn.ReLU(
                inplace=True
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.block(x)


class LightweightUNet(nn.Module):
    """
    Lightweight U-Net for 64x64 path-preference prediction.

    Input:
        (B, 5, 64, 64)

    Output:
        (B, 1, 64, 64)

    The model returns raw logits. A sigmoid is applied only when
    probabilities are required for visualization or inference.
    """

    def __init__(
        self,
        in_channels: int = 5,
        out_channels: int = 1,
        base_channels: int = 16,
    ) -> None:
        super().__init__()

        # ---------------------------------------------------------
        # Encoder
        # ---------------------------------------------------------
        self.encoder1 = DoubleConv(
            in_channels,
            base_channels,
        )

        self.pool1 = nn.MaxPool2d(
            kernel_size=2
        )

        self.encoder2 = DoubleConv(
            base_channels,
            base_channels * 2,
        )

        self.pool2 = nn.MaxPool2d(
            kernel_size=2
        )

        self.encoder3 = DoubleConv(
            base_channels * 2,
            base_channels * 4,
        )

        self.pool3 = nn.MaxPool2d(
            kernel_size=2
        )

        # ---------------------------------------------------------
        # Bottleneck
        # ---------------------------------------------------------
        self.bottleneck = DoubleConv(
            base_channels * 4,
            base_channels * 8,
        )

        # ---------------------------------------------------------
        # Decoder
        # ---------------------------------------------------------
        self.up3 = nn.ConvTranspose2d(
            base_channels * 8,
            base_channels * 4,
            kernel_size=2,
            stride=2,
        )

        self.decoder3 = DoubleConv(
            base_channels * 8,
            base_channels * 4,
        )

        self.up2 = nn.ConvTranspose2d(
            base_channels * 4,
            base_channels * 2,
            kernel_size=2,
            stride=2,
        )

        self.decoder2 = DoubleConv(
            base_channels * 4,
            base_channels * 2,
        )

        self.up1 = nn.ConvTranspose2d(
            base_channels * 2,
            base_channels,
            kernel_size=2,
            stride=2,
        )

        self.decoder1 = DoubleConv(
            base_channels * 2,
            base_channels,
        )

        # ---------------------------------------------------------
        # Output layer
        # ---------------------------------------------------------
        self.output_layer = nn.Conv2d(
            base_channels,
            out_channels,
            kernel_size=1,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        # Encoder
        encoder1 = self.encoder1(x)

        encoder2 = self.encoder2(
            self.pool1(encoder1)
        )

        encoder3 = self.encoder3(
            self.pool2(encoder2)
        )

        # Bottleneck
        bottleneck = self.bottleneck(
            self.pool3(encoder3)
        )

        # Decoder level 3
        decoder3 = self.up3(
            bottleneck
        )

        decoder3 = torch.cat(
            [
                decoder3,
                encoder3,
            ],
            dim=1,
        )

        decoder3 = self.decoder3(
            decoder3
        )

        # Decoder level 2
        decoder2 = self.up2(
            decoder3
        )

        decoder2 = torch.cat(
            [
                decoder2,
                encoder2,
            ],
            dim=1,
        )

        decoder2 = self.decoder2(
            decoder2
        )

        # Decoder level 1
        decoder1 = self.up1(
            decoder2
        )

        decoder1 = torch.cat(
            [
                decoder1,
                encoder1,
            ],
            dim=1,
        )

        decoder1 = self.decoder1(
            decoder1
        )

        logits = self.output_layer(
            decoder1
        )

        return logits