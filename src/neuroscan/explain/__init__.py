"""Visual explainability.

Report section 2.2 identifies the black-box problem as a live barrier to
clinical adoption, and section 2.5 notes that explainability techniques "are
still rarely integrated into fully deployable clinical systems". Grad-CAM is
therefore part of the delivered application, not an offline analysis step:
every prediction the web interface shows is accompanied by the heatmap that
produced it.
"""

from neuroscan.explain.gradcam import (
    GradCAM,
    GradCAMResult,
    explain_prediction,
    overlay_heatmap,
)

__all__ = ["GradCAM", "GradCAMResult", "explain_prediction", "overlay_heatmap"]
