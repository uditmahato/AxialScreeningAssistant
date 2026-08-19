"""Shared interface for every Axial Screening Assistant classifier.

Fixing this contract is what lets the training harness, the Grad-CAM
implementation and the Flask application treat a scratch-built CNN and a
pre-trained EfficientNet identically. In particular, ``gradcam_target_layer``
is part of the model's own definition rather than something the explainability
code guesses at by walking the module tree - a guess that silently produces a
meaningless heatmap when it picks the wrong layer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class BaseClassifier(nn.Module, ABC):
    """Common behaviour for the comparative architectures.

    Args:
        num_classes: Output dimensionality. Two for the binary task.
        class_names: Ordered class labels; index ``i`` is logit ``i``.
    """

    def __init__(self, num_classes: int, class_names: list[str] | None = None) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError(f"num_classes must be at least 2, got {num_classes}")
        self.num_classes = num_classes
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]
        if len(self.class_names) != num_classes:
            raise ValueError(
                f"class_names has {len(self.class_names)} entries but num_classes is {num_classes}"
            )

    # -- architecture-specific ------------------------------------------------

    @property
    @abstractmethod
    def architecture_name(self) -> str:
        """Registry key, e.g. ``'efficientnet_b0'``."""

    @abstractmethod
    def gradcam_target_layer(self) -> nn.Module:
        """The convolutional layer Grad-CAM should hook.

        Must be the last layer that still carries spatial structure - typically
        the final convolutional block before global pooling. Hooking anything
        after pooling yields a 1x1 map and therefore a uniform, useless heatmap.
        """

    @abstractmethod
    def backbone_modules(self) -> list[nn.Module]:
        """Feature-extraction modules, ordered shallow to deep.

        Used by :meth:`unfreeze_last` to reopen the deepest blocks during
        fine-tuning.
        """

    @abstractmethod
    def head_modules(self) -> list[nn.Module]:
        """The randomly-initialised classifier head."""

    # -- shared -------------------------------------------------------------

    def freeze_backbone(self) -> int:
        """Freeze all feature-extraction parameters.

        Stage 1 of the two-stage schedule. The head starts random, so its early
        gradients are large; letting those flow into pre-trained features
        destroys exactly the representations transfer learning was meant to
        reuse.

        Returns:
            Number of parameters frozen.
        """
        frozen = 0
        for module in self.backbone_modules():
            for param in module.parameters():
                if param.requires_grad:
                    param.requires_grad = False
                    frozen += param.numel()
        return frozen

    def unfreeze_all(self) -> int:
        """Make every parameter trainable. Returns the count unfrozen."""
        unfrozen = 0
        for param in self.parameters():
            if not param.requires_grad:
                param.requires_grad = True
                unfrozen += param.numel()
        return unfrozen

    def unfreeze_last(self, n_layers: int) -> int:
        """Unfreeze the deepest ``n_layers`` parameterised backbone layers.

        Stage 2. Deep layers encode task-specific structure and benefit most
        from adaptation; shallow layers encode edges and textures that transfer
        essentially unchanged from ImageNet to MRI. Passing a value at or above
        the layer count unfreezes the whole backbone.

        Returns:
            Number of parameters unfrozen.
        """
        if n_layers <= 0:
            return 0

        parameterised: list[nn.Module] = []
        for module in self.backbone_modules():
            for submodule in module.modules():
                if any(p is not None for p in submodule.parameters(recurse=False)):
                    parameterised.append(submodule)

        unfrozen = 0
        for submodule in parameterised[-n_layers:]:
            for param in submodule.parameters(recurse=False):
                if not param.requires_grad:
                    param.requires_grad = True
                    unfrozen += param.numel()
        return unfrozen

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def parameter_groups(self, lr_backbone: float, lr_head: float) -> list[dict]:
        """Discriminative learning rates for backbone versus head.

        The head is new and needs to move quickly; the backbone is pre-trained
        and needs only gentle adjustment. Applying one rate to both either
        starves the head or wrecks the features.
        """
        backbone_params = [
            p for module in self.backbone_modules() for p in module.parameters() if p.requires_grad
        ]
        head_params = [
            p for module in self.head_modules() for p in module.parameters() if p.requires_grad
        ]

        groups: list[dict] = []
        if backbone_params:
            groups.append({"params": backbone_params, "lr": lr_backbone, "name": "backbone"})
        if head_params:
            groups.append({"params": head_params, "lr": lr_head, "name": "head"})
        return groups

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Class probabilities for a batch. Sets eval mode as a side effect."""
        self.eval()
        return torch.softmax(self(x), dim=1)

    def describe(self) -> dict[str, object]:
        """Summary recorded in the run manifest and the comparison table."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "architecture": self.architecture_name,
            "num_classes": self.num_classes,
            "class_names": list(self.class_names),
            "total_parameters": total,
            "trainable_parameters": trainable,
            "frozen_parameters": total - trainable,
            "size_mb": round(total * 4 / 1024**2, 2),  # float32
        }


__all__ = ["BaseClassifier"]
