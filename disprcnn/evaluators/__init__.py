# from disprcnn.metric.accuracy import accuracy
from .kittiobj import *
from .yolact import YolactEvaluator


def build_evaluators(cfg):
    evaluators = []
    for e in cfg.test.evaluators:
        evaluators.append(EVALUATORS[e](cfg))
    return evaluators
