"""Axial Screening Assistant.

Brain MRI screening support and bilingual clinical advisory for
resource-limited healthcare settings.

The package is organised into loosely-coupled subsystems so that each can be
developed, tested and swapped independently:

    neuroscan.data        MRI ingestion, CLAHE preprocessing, leakage-safe splits
    neuroscan.models      Baseline CNN, VGG16 and EfficientNetB0 classifiers
    neuroscan.training    Two-stage transfer-learning harness and cross-validation
    neuroscan.evaluation  Metrics, curves and cross-architecture comparison
    neuroscan.explain     Grad-CAM visual explanations
    neuroscan.rag         Knowledge corpus, FAISS retrieval and advisory generation
    neuroscan.chatbot     Bilingual (English / Nepali) conversational interface
    neuroscan.reporting   Downloadable PDF clinical reports
    neuroscan.web         Flask application tying the subsystems together

IMPORTANT - CLINICAL SAFETY
This software is a decision-support prototype. It does not provide a
diagnosis, and every user-facing output must carry the disclaimer defined in
``neuroscan.safety.DISCLAIMER_EN`` / ``DISCLAIMER_NE``.
"""

__version__ = "1.0.0"

__all__ = ["__version__"]
