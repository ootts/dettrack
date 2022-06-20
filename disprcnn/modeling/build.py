import loguru

from disprcnn.modeling.models.fusion.fusion import DepthFusion
from disprcnn.modeling.models.fusion.fusion_onetime import DepthFusionOneTime
from disprcnn.modeling.models.nerf.core import NeRFWarpper
from disprcnn.modeling.models.nsg.core import NsgWarpper
from disprcnn.modeling.models.nsg2.core import NsgWarpper2
from disprcnn.modeling.models.nsg_sdf.core import NsgSDFWarpper
from disprcnn.modeling.models.slot_attention.core import SlotAttentionAutoEncoder
from disprcnn.modeling.models.star.core import STaR
from disprcnn.modeling.models.starmo.starmo import STaRMo
from disprcnn.modeling.models.star_ps.core import STaRPs
from disprcnn.modeling.models.savi.savi import Savi
# from disprcnn.modeling.models.disprcnn.nds import NdsWarpper
from disprcnn.modeling.models.star_sdf.star_sdf import STaRSdf
from disprcnn.modeling.models.star_sdf.starsdfmo import STaRSdfMo
from disprcnn.modeling.models.star_sdf.starsdfmouni import STaRSdfMoUni
from disprcnn.modeling.models.star_mm.starmm import STaRMM
from disprcnn.modeling.models.volsdf.volsdf import VolSDFWarpper
from disprcnn.modeling.models.neural_diff.neuraldiff import NeuralDiffWarpper
from disprcnn.modeling.models.barf.barf import BaRF
from disprcnn.modeling.models.volsdf_surface.surface import ImplicitSurface
from disprcnn.modeling.models.fbs.fbs import FlowBasedSegm

_META_ARCHITECTURES = {'NeRF': NeRFWarpper,
                       'SlotAttentionAutoEncoder': SlotAttentionAutoEncoder,
                       'NeuralSceneGraph': NsgWarpper,
                       'NeuralSceneGraph2': NsgWarpper2,
                       'NeuralSceneGraphSDF': NsgSDFWarpper,
                       'STaR': STaR,
                       'Savi': Savi,
                       # 'Nds': NdsWarpper,
                       'STaRSdf': STaRSdf,
                       'STaRSdfMo': STaRSdfMo,
                       'STaRSdfMoUni': STaRSdfMoUni,
                       'STaRPs': STaRPs,
                       'STaRMM': STaRMM,
                       'DepthFusion': DepthFusion,
                       'DepthFusionOneTime': DepthFusionOneTime,
                       'STaRMo': STaRMo,
                       'VolSDF': VolSDFWarpper,
                       'NeuralDiff': NeuralDiffWarpper,
                       'BaRF': BaRF,
                       'ImplicitSurface': ImplicitSurface,
                       'FlowBasedSegm': FlowBasedSegm
                       }


def build_model(cfg):
    print("building model...", end='\r')
    meta_arch = _META_ARCHITECTURES[cfg.model.meta_architecture]
    model = meta_arch(cfg)
    # loguru.logger.info("Done.")
    return model
