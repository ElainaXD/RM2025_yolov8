# Ultralytics YOLO 🚀, GPL-3.0 license

__version__ = "8.0.29"

from .yolo.engine.model import YOLO
from .yolo.utils import ops
from .yolo.utils.checks import check_yolo as checks

__all__ = ["__version__", "YOLO", "hub", "checks"]  # allow simpler import
