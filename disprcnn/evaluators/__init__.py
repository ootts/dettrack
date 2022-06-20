from ..registry import EVALUATORS
from .starsdfmo import starsdfmo_pose_eval, starsdfmo_mask_eval
from .fusion import fusion_mask, fusion_objpose
from .fusionot import fusionot_mask, fusionot_objpose
from .starsdf import starsdf_mask_eval, starsdf_pose_eval
from .starsdfmouni import starsdfmouni_mask_eval, starsdfmouni_pose_eval

from .build import *


def build_evaluators(cfg):
    evaluators = []
    for e in cfg.test.evaluators:
        evaluators.append(EVALUATORS[e](cfg))
    return evaluators
