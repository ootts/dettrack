from ..registry import VISUALIZERS
from .yolact import YolactVisualizer
from .yolact_tracking import YolactTrackingVisualizer


def build_visualizer(cfg):
    return VISUALIZERS[cfg.test.visualizer](cfg)
