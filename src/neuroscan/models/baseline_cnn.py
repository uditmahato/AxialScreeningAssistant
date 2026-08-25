"""Custom CNN trained from scratch - the control arm of the comparison.

Its purpose is to quantify what transfer learning actually buys on this
dataset. Without it, a strong EfficientNetB0 result could just as easily be
evidence that the task is easy as evidence that pre-training helps
(Project Design 4.2: "provides a balance between the modern architectures and
a base model").

Design choices follow from having only a few hundred to a few thousand images:

* **Four blocks, ~1.2M parameters.** Deeper or wider overfits before it
  generalises at this scale.
* **BatchNorm after every convolution.** Without pre-trained statistics, a
  scratch network needs the normalisation to train stably at a useful rate.
* **Global average pooling instead of a flatten.** A flatten at 14x14x256 would
  produce a 50k-wide feature vector and a head with more parameters than the
  entire feature extractor - the classic overfitting trap on small datasets.
"""

from __future__ import annotations

import torch
from torch import nn

from neuroscan.models.base import BaseClassifier


class ConvBlock(nn.Module):
    """Two Conv-BN-ReLU units followed by max pooling.

    The paired convolutions give each block a 5x5 effective receptive field at
    the parameter cost of two 3x3 kernels, and insert an extra non-linearity.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class BaselineCNN(BaseClassifier):
    """A four-block convolutional classifier trained from random initialisation.

    Args:
        num_classes: Output dimensionality.
        class_names: Ordered class labels.
        dropout: Dropout probability before the final linear layer.
        base_channels: Channel width of the first block; each block doubles it.
    """

    def __init__(
        self,
        num_classes: int = 2,
        class_names: list[str] | None = None,
        *,
        dropout: float = 0.5,
        base_channels: int = 32,
    ) -> None:
        super().__init__(num_classes, class_names)

        c1 = base_channels          # 32   224 -> 112
        c2 = base_channels * 2      # 64   112 -> 56
        c3 = base_channels * 4      # 128   56 -> 28
        c4 = base_channels * 8      # 256   28 -> 14

        self.block1 = ConvBlock(3, c1)
        self.block2 = ConvBlock(c1, c2)
        self.block3 = ConvBlock(c2, c3)
        self.block4 = ConvBlock(c3, c4)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(c4, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(128, num_classes),
        )

        self._initialise_weights()

    def _initialise_weights(self) -> None:
        """He initialisation for the convolutions.

        Correct for ReLU: it preserves activation variance through depth, where
        the default uniform initialisation lets it decay and stalls training in
        the deeper blocks.
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.zeros_(module.bias)

    @property
    def architecture_name(self) -> str:
        return "baseline_cnn"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool(x)
        return self.classifier(x)

    def gradcam_target_layer(self) -> nn.Module:
        # The last conv block still carries a 14x14 spatial grid at 224x224
        # input - coarse, but enough to localise a mass.
        return self.block4

    def backbone_modules(self) -> list[nn.Module]:
        return [self.block1, self.block2, self.block3, self.block4]

    def head_modules(self) -> list[nn.Module]:
        return [self.classifier]


__all__ = ["BaselineCNN", "ConvBlock"]
