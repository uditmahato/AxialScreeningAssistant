"""Classifier architectures for the comparative study (Project Design 4.1).

Three models are compared under an identical data pipeline and training
schedule, so any difference in the results is attributable to the architecture
rather than to preprocessing or optimisation:

    baseline_cnn      trained from scratch - the control arm
    vgg16             transfer learning, deep sequential features
    efficientnet_b0   transfer learning, compound-scaled and efficient

All three subclass :class:`BaseClassifier`, which fixes the interface the
training harness, Grad-CAM and the web application depend on.
"""

from neuroscan.models.base import BaseClassifier
from neuroscan.models.baseline_cnn import BaselineCNN
from neuroscan.models.factory import ARCHITECTURES, build_model, load_checkpoint, save_checkpoint
from neuroscan.models.transfer import EfficientNetB0Classifier, VGG16Classifier

__all__ = [
    "ARCHITECTURES",
    "BaseClassifier",
    "BaselineCNN",
    "EfficientNetB0Classifier",
    "VGG16Classifier",
    "build_model",
    "load_checkpoint",
    "save_checkpoint",
]
