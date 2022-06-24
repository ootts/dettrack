import loguru

from disprcnn.modeling.models.yolact.yolact import YolactWrapper
from disprcnn.modeling.models.yolact.yolact_tracking import YolactTracking

_META_ARCHITECTURES = {'Yolact': YolactWrapper,
                       'YolactTracking': YolactTracking
                       }


def build_model(cfg):
    print("building model...", end='\r')
    meta_arch = _META_ARCHITECTURES[cfg.model.meta_architecture]
    model = meta_arch(cfg)
    loguru.logger.info("Done.")
    return model
