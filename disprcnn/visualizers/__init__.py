from ..registry import VISUALIZERS
from .yolact import YolactVisualizer


def build_visualizer(cfg):
    return VISUALIZERS[cfg.test.visualizer](cfg)
