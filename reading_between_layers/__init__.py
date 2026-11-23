"""Reading Between the Layers: layerwise representation quality and the Stable Geometry Index.

Quick start::

    from reading_between_layers import sgi
    layer = sgi.select_layer(per_layer_metrics)   # columns: layer_index + sgi.SGI_METRICS

The GPU pipeline lives in ``pipeline/``; CPU-side reproduction of the paper's
tables and figures lives in ``scripts/reproduce_*.py``.
"""

from .sgi import SGI_METRICS, select_layer, select_layers, sgi_score, sgi_terms

__version__ = "0.1.0"
__all__ = ["SGI_METRICS", "select_layer", "select_layers", "sgi_score", "sgi_terms"]
