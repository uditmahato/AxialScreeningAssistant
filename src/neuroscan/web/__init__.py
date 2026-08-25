"""Flask web application.

Integrates MRI upload, classification, Grad-CAM visualisation, RAG advisory,
the bilingual chatbot and PDF report generation into one interface.
"""

from neuroscan.web.app import create_app

__all__ = ["create_app"]
