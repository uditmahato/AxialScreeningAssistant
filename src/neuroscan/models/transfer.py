"""Transfer-learning architectures: VGG16 and EfficientNetB0.

Both load ImageNet weights and replace the 1000-class head with a small
task-specific one. The justification is in Project Design 4.2: transfer
learning outperforms training from scratch on small medical image datasets
because the low-level filters - edges, textures, intensity gradients - are
shared between natural images and MRI, and only the deeper, semantic layers
need to be re-learned.

**Head design.** Both heads global-average-pool to a compact feature vector
before the linear layers. VGG16's original head flattens a 7x7x512 map into
25,088 features and passes it through two 4,096-wide layers - about 120M
parameters, on a dataset of a few hundred images. Such a head memorises the
training set long before the features learn anything transferable. Pooling
first reduces the head to well under a million parameters and is the standard
modern remedy; the cost is losing coarse spatial layout, which for a
whole-slice normal/abnormal decision is not information the classifier needs.
"""

from __future__ import annotations

import torch
from torch import nn

from neuroscan.models.base import BaseClassifier
from neuroscan.utils import get_logger

log = get_logger("models.transfer")

#: Index of the ReLU following VGG16's final convolution, within
#: ``torchvision``'s ``features`` Sequential. Named rather than inlined because
#: two places depend on it staying consistent.
GRADCAM_LAYER_INDEX = 29


def _build_pooled_head(in_features: int, num_classes: int, dropout: float) -> nn.Sequential:
    """Compact classifier head shared by both transfer architectures.

    Identical across models so the comparison isolates the backbone rather than
    conflating it with head capacity.
    """
    return nn.Sequential(
        nn.Flatten(),
        nn.Dropout(dropout),
        nn.Linear(in_features, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout / 2),
        nn.Linear(256, num_classes),
    )


def _init_head(head: nn.Module) -> None:
    """Initialise the newly-created head layers."""
    for module in head.modules():
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, 0, 0.01)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)


class VGG16Classifier(BaseClassifier):
    """VGG16 with ImageNet features and a pooled classification head.

    VGG16 is included as the established, widely-cited transfer baseline. It is
    the heaviest of the three at ~14.7M backbone parameters with no residual
    connections, which is precisely why the comparison against EfficientNetB0
    is informative for a deployment target that may have no GPU at all.
    """

    def __init__(
        self,
        num_classes: int = 2,
        class_names: list[str] | None = None,
        *,
        pretrained: bool = True,
        dropout: float = 0.5,
    ) -> None:
        super().__init__(num_classes, class_names)
        from torchvision import models

        weights = models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        if pretrained:
            log.info("Loading VGG16 ImageNet weights (IMAGENET1K_V1)")
        backbone = models.vgg16(weights=weights)

        self.features = backbone.features           # 13 conv layers -> 512x7x7
        self.pool = nn.AdaptiveAvgPool2d(1)         # -> 512x1x1
        self.classifier = _build_pooled_head(512, num_classes, dropout)
        _init_head(self.classifier)

        # torchvision builds VGG16 with in-place ReLUs, which saves memory but
        # makes the Grad-CAM target unhookable: a full backward hook on a
        # module whose output is modified in place raises
        # "Output 0 of BackwardHookFunction is a view and is being modified
        # inplace". Disabling it on the hooked layer alone costs one 14x14x512
        # activation tensor and keeps every other ReLU in place.
        target = self.features[GRADCAM_LAYER_INDEX]
        if isinstance(target, nn.ReLU):
            target.inplace = False

    @property
    def architecture_name(self) -> str:
        return "vgg16"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)

    def gradcam_target_layer(self) -> nn.Module:
        # The ReLU after the final 3x3 convolution, producing a 14x14x512 map.
        # The following MaxPool would halve that to 7x7 and blur the
        # localisation for no benefit.
        return self.features[GRADCAM_LAYER_INDEX]

    def backbone_modules(self) -> list[nn.Module]:
        return [self.features]

    def head_modules(self) -> list[nn.Module]:
        return [self.classifier]


class EfficientNetB0Classifier(BaseClassifier):
    """EfficientNetB0 with ImageNet features and a pooled classification head.

    The intended production model. Compound scaling gives it roughly VGG16's
    accuracy at a third of the parameters and a fraction of the FLOPs, which is
    what makes CPU-only inference viable in a district hospital
    (Project Design 4.2, and section 2.3 on low-resource deployment).
    """

    def __init__(
        self,
        num_classes: int = 2,
        class_names: list[str] | None = None,
        *,
        pretrained: bool = True,
        dropout: float = 0.3,
    ) -> None:
        super().__init__(num_classes, class_names)
        from torchvision import models

        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        if pretrained:
            log.info("Loading EfficientNetB0 ImageNet weights (IMAGENET1K_V1)")
        backbone = models.efficientnet_b0(weights=weights)

        self.features = backbone.features           # -> 1280x7x7
        self.pool = nn.AdaptiveAvgPool2d(1)         # -> 1280x1x1
        self.classifier = _build_pooled_head(1280, num_classes, dropout)
        _init_head(self.classifier)

    @property
    def architecture_name(self) -> str:
        return "efficientnet_b0"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)

    def gradcam_target_layer(self) -> nn.Module:
        # features[-1] is the final Conv2dNormActivation (1280 channels, 7x7).
        # Coarse, but it is the deepest layer retaining spatial structure and
        # is the conventional Grad-CAM hook point for this family.
        return self.features[-1]

    def backbone_modules(self) -> list[nn.Module]:
        return [self.features]

    def head_modules(self) -> list[nn.Module]:
        return [self.classifier]


__all__ = ["EfficientNetB0Classifier", "VGG16Classifier"]
