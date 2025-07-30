"""
OpiClaw: Deep-Sea Panoptic Segmentation
Core model implementation package
"""

__version__ = "1.0.0"
__author__ = "Matthew Caldwell"
__email__ = "matt@hyperbid.us"

from .models import PanopticOpiClaw
from .utils import project_to_polar_bev, generate_marine_scene

__all__ = ["PanopticOpiClaw", "project_to_polar_bev", "generate_marine_scene"]
