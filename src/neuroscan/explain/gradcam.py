"""Grad-CAM: gradient-weighted class activation mapping.

The method, briefly: hook the last convolutional layer to capture both its
forward activations ``A`` and the gradient of the target class score with
respect to them. Global-average-pool those gradients to get one importance
weight per channel, take the weighted sum of the activation maps, and apply
ReLU. The ReLU matters - negative contributions are evidence *against* the
class, and including them would produce a map highlighting regions that argued
for the opposite conclusion.

Two implementation points that are easy to get wrong:

**Grad-CAM must run in float32.** Under autocast the activations are float16,
and the gradients of a confident prediction underflow to zero, yielding a blank
heatmap. Every forward and backward pass here is explicitly full precision.

**Hooks must be removed.** A model that keeps accumulating forward hooks leaks
memory across requests and, worse, retains the computation graph. The hooks are
managed by a context manager so they are released even if generation raises.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
import torch
from torch import nn

from neuroscan.utils import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from neuroscan.config import PreprocessingConfig
    from neuroscan.models.base import BaseClassifier

log = get_logger("explain.gradcam")


@dataclass
class GradCAMResult:
    """One explanation."""

    heatmap: np.ndarray          # float32 (H, W) in [0, 1], input resolution
    overlay: np.ndarray          # uint8 (H, W, 3) RGB, heatmap over the scan
    target_class: int
    target_class_name: str
    confidence: float
    #: Fraction of the frame the model attended to, at half-maximum activation.
    #: A very high value means diffuse, unfocused attention - a useful signal
    #: that the heatmap should not be over-interpreted.
    focus_ratio: float
    #: Centre of mass of the activation, normalised to [0, 1] as (x, y).
    peak_location: tuple[float, float]
    #: The map carried no activation at all. Distinct from diffuse: a blank
    #: map is the known failure mode (gradients underflowing under autocast,
    #: an in-place target ReLU), and the interface must say "no map could be
    #: generated" rather than implying the model saw nothing anywhere.
    is_blank: bool = False

    def is_diffuse(self, threshold: float = 0.45) -> bool:
        """Whether attention is spread widely rather than localised.

        Surfaced in the UI so a diffuse map is presented as "the model did not
        localise a specific region" instead of implying a precise finding.
        """
        return self.focus_ratio > threshold


class GradCAM:
    """Grad-CAM generator bound to a model's designated target layer.

    Args:
        model: A :class:`BaseClassifier`; its ``gradcam_target_layer`` is used.
        target_layer: Optional override for experimentation.
    """

    def __init__(self, model: BaseClassifier, target_layer: nn.Module | None = None) -> None:
        self.model = model
        self.target_layer = target_layer or model.gradcam_target_layer()
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def __enter__(self) -> GradCAM:
        self._register()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.remove_hooks()

    def _register(self) -> None:
        def forward_hook(_module, _inputs, output) -> None:
            self._activations = output

        def backward_hook(_module, _grad_input, grad_output) -> None:
            self._gradients = grad_output[0]

        self._handles.append(self.target_layer.register_forward_hook(forward_hook))
        # full_backward_hook fires reliably for modules with multiple outputs,
        # unlike the deprecated register_backward_hook.
        self._handles.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._activations = None
        self._gradients = None

    def generate(
        self,
        input_tensor: torch.Tensor,
        *,
        target_class: int | None = None,
    ) -> tuple[np.ndarray, int, float]:
        """Produce a heatmap for one image.

        Args:
            input_tensor: Shape ``(1, 3, H, W)``, already normalised.
            target_class: Class to explain. Defaults to the predicted class.

        Returns:
            ``(heatmap, target_class, confidence)`` with the heatmap at the
            target layer's spatial resolution, normalised to ``[0, 1]``.
        """
        if input_tensor.dim() != 4 or input_tensor.size(0) != 1:
            raise ValueError(f"Expected a single image of shape (1, 3, H, W), got {tuple(input_tensor.shape)}")

        if not self._handles:
            raise RuntimeError("Hooks are not registered - use GradCAM as a context manager")

        was_training = self.model.training
        self.model.eval()

        # Explicitly full precision: see the module docstring.
        with torch.amp.autocast("cuda", enabled=False):
            input_tensor = input_tensor.float().requires_grad_(True)
            logits = self.model(input_tensor)
            probabilities = torch.softmax(logits, dim=1)

            if target_class is None:
                target_class = int(logits.argmax(dim=1).item())
            confidence = float(probabilities[0, target_class].detach())

            self.model.zero_grad(set_to_none=True)
            logits[0, target_class].backward()

        if was_training:
            self.model.train()

        if self._activations is None or self._gradients is None:
            raise RuntimeError(
                "Grad-CAM captured no activations or gradients. The target layer may not "
                "participate in the forward pass."
            )

        activations = self._activations[0].detach()   # (C, h, w)
        gradients = self._gradients[0].detach()       # (C, h, w)

        # One importance weight per channel: the mean gradient over space.
        weights = gradients.mean(dim=(1, 2), keepdim=True)
        cam = (weights * activations).sum(dim=0)
        cam = torch.relu(cam)

        cam_np = cam.cpu().numpy().astype(np.float32)
        peak = float(cam_np.max())
        if peak <= 1e-8:
            # Legitimate outcome: no positive evidence for this class anywhere.
            # A flat map is the honest representation, not an error.
            log.warning(
                "Grad-CAM produced a uniformly zero map for class %d - the model found no "
                "positively-contributing region.", target_class,
            )
            return np.zeros_like(cam_np), target_class, confidence

        cam_np = cam_np / peak
        return cam_np, target_class, confidence


def _resize_heatmap(heatmap: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Upsample a coarse CAM to image resolution.

    Bicubic can overshoot outside ``[0, 1]`` and create bright rings around
    activation peaks that look like structure but are pure interpolation
    artefact, so bilinear is used and the result is clipped.
    """
    resized = cv2.resize(heatmap, size, interpolation=cv2.INTER_LINEAR)
    return np.clip(resized, 0.0, 1.0)


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    *,
    alpha: float = 0.40,
    colormap: int = cv2.COLORMAP_JET,
    suppress_below: float = 0.25,
) -> np.ndarray:
    """Blend a heatmap over the scan.

    Args:
        image: RGB ``uint8`` base image.
        heatmap: Float map in ``[0, 1]``, any resolution.
        alpha: Peak opacity of the colour layer.
        colormap: OpenCV colormap. JET is conventional for Grad-CAM.
        suppress_below: Activations below this are made fully transparent.
            Without it the low-level noise floor tints the entire brain, which
            reads to a clinician as widespread abnormality.

    Returns:
        RGB ``uint8`` overlay at the base image's resolution.
    """
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    height, width = image.shape[:2]
    resized = _resize_heatmap(heatmap, (width, height))

    coloured = cv2.applyColorMap((resized * 255).astype(np.uint8), colormap)
    coloured = cv2.cvtColor(coloured, cv2.COLOR_BGR2RGB).astype(np.float32)

    # Opacity scales with activation, so weak regions stay near-transparent
    # and the strongest region reaches full alpha.
    mask = np.clip((resized - suppress_below) / max(1e-6, 1.0 - suppress_below), 0.0, 1.0)
    mask = (mask * alpha)[..., None]

    blended = image.astype(np.float32) * (1.0 - mask) + coloured * mask
    return np.clip(blended, 0, 255).astype(np.uint8)


def _focus_statistics(heatmap: np.ndarray) -> tuple[float, tuple[float, float]]:
    """Measure how localised a heatmap is, and where its mass sits."""
    if heatmap.size == 0 or heatmap.max() <= 0:
        return 1.0, (0.5, 0.5)

    # Fraction of pixels at or above half maximum - a simple, interpretable
    # proxy for spread that needs no threshold tuning.
    focus_ratio = float((heatmap >= 0.5).mean())

    total = heatmap.sum()
    if total <= 0:
        return focus_ratio, (0.5, 0.5)

    ys, xs = np.mgrid[0 : heatmap.shape[0], 0 : heatmap.shape[1]]
    cx = float((xs * heatmap).sum() / total) / max(heatmap.shape[1] - 1, 1)
    cy = float((ys * heatmap).sum() / total) / max(heatmap.shape[0] - 1, 1)
    return focus_ratio, (round(cx, 4), round(cy, 4))


def explain_prediction(
    model: BaseClassifier,
    input_tensor: torch.Tensor,
    display_image: np.ndarray,
    *,
    target_class: int | None = None,
    alpha: float = 0.40,
) -> GradCAMResult:
    """Generate a complete explanation for one scan.

    Args:
        model: Trained classifier.
        input_tensor: Normalised tensor, shape ``(1, 3, H, W)``.
        display_image: The standardised ``uint8`` RGB image the tensor came
            from, used as the overlay base so the heatmap aligns with exactly
            what the model saw.
        target_class: Class to explain; defaults to the prediction.
        alpha: Overlay opacity.

    Returns:
        A populated :class:`GradCAMResult`.
    """
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)

    with GradCAM(model) as cam:
        heatmap, resolved_class, confidence = cam.generate(
            input_tensor, target_class=target_class
        )

    height, width = display_image.shape[:2]
    full_heatmap = _resize_heatmap(heatmap, (width, height))
    overlay = overlay_heatmap(display_image, full_heatmap, alpha=alpha)
    focus_ratio, peak = _focus_statistics(full_heatmap)

    class_names = getattr(model, "class_names", [])
    class_name = (
        class_names[resolved_class] if resolved_class < len(class_names) else str(resolved_class)
    )

    result = GradCAMResult(
        heatmap=full_heatmap,
        overlay=overlay,
        target_class=resolved_class,
        target_class_name=class_name,
        confidence=confidence,
        focus_ratio=focus_ratio,
        peak_location=peak,
        is_blank=bool(full_heatmap.size == 0 or float(full_heatmap.max()) <= 0.0),
    )

    if result.is_blank:
        log.error(
            "Grad-CAM produced a blank map for class %s - the overlay carries no "
            "activation and must not be presented as an explanation.", resolved_class,
        )
    if result.is_diffuse():
        log.info(
            "Grad-CAM attention is diffuse (focus_ratio=%.2f) - the explanation should be "
            "presented as non-localising.", focus_ratio,
        )
    return result


def save_explanation(
    result: GradCAMResult,
    display_image: np.ndarray,
    out_path: Path,
    *,
    side_by_side: bool = True,
) -> Path:
    """Write the explanation to disk.

    The side-by-side form pairs the original with the overlay, which is what a
    clinician needs: the heatmap alone gives no way to check whether the
    highlighted region corresponds to real anatomy.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if side_by_side:
        separator = np.full((display_image.shape[0], 8, 3), 255, dtype=np.uint8)
        canvas = np.hstack([display_image, separator, result.overlay])
    else:
        canvas = result.overlay

    cv2.imwrite(str(out_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    return out_path


def explain_from_file(
    model: BaseClassifier,
    image_path: Path,
    pre_cfg: PreprocessingConfig,
    *,
    target_class: int | None = None,
) -> GradCAMResult:
    """Convenience wrapper: load a file, preprocess it, and explain it."""
    from neuroscan.data.preprocessing import preprocess_for_inference

    tensor, display = preprocess_for_inference(image_path, pre_cfg)
    return explain_prediction(model, tensor, display, target_class=target_class)


__all__ = [
    "GradCAM",
    "GradCAMResult",
    "explain_from_file",
    "explain_prediction",
    "overlay_heatmap",
    "save_explanation",
]
