"""UNet3D stub for FedMed.

This module defines a lightweight, import-safe stub for a MONAI/PyTorch-style
3D U-Net. It deliberately avoids importing torch at module-import time so the
package can be imported in environments without PyTorch installed. Call
build_torch_model() at runtime if torch is available to obtain a concrete
nn.Module.
"""
from typing import Sequence, Tuple


class UNet3D:
    """Lightweight, dependency-safe 3D U-Net stub.

    Parameters
    - in_channels: number of input channels (e.g., 1 for single-channel MRI)
    - out_channels: number of output channels (e.g., number of segmentation classes)
    - features: tuple with feature sizes per downsampling level

    Usage:
        net = UNet3D()
        # If PyTorch is installed and you want an actual nn.Module:
        model = net.build_torch_model()
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1, features: Sequence[int] = (32, 64, 128)):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = tuple(features)

    def summary(self) -> str:
        """Return a short textual summary describing the intended architecture."""
        lines = [
            "UNet3D (stub)",
            f"  in_channels: {self.in_channels}",
            f"  out_channels: {self.out_channels}",
            f"  features: {self.features}",
            "  Note: call build_torch_model() to get a real nn.Module when PyTorch is available",
        ]
        return "\n".join(lines)

    def build_torch_model(self):
        """Build and return a very small PyTorch nn.Module implementing the
        high-level UNet interface. This method imports torch lazily and raises
        a RuntimeError if torch is not available.
        """
        try:
            import torch
            import torch.nn as nn
        except Exception as exc:
            raise RuntimeError("PyTorch is required to build the real model") from exc

        # A highly simplified placeholder: just a few conv layers to act as a
        # drop-in for demonstration and unit tests.
        class SimpleUNet(nn.Module):
            def __init__(self, in_ch, out_ch, feats):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Conv3d(in_ch, feats[0], kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool3d(2),
                )
                self.bottleneck = nn.Sequential(
                    nn.Conv3d(feats[0], feats[1], kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                )
                self.decoder = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
                    nn.Conv3d(feats[1], out_ch, kernel_size=1),
                )

            def forward(self, x):
                x = self.encoder(x)
                x = self.bottleneck(x)
                x = self.decoder(x)
                return x

        return SimpleUNet(self.in_channels, self.out_channels, list(self.features))
