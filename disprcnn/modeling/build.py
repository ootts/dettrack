import loguru

from nds.modeling.models.fusion.fusion import DepthFusion
from nds.modeling.models.fusion.fusion_onetime import DepthFusionOneTime
from nds.modeling.models.nerf.core import NeRFWarpper
from nds.modeling.models.nsg.core import NsgWarpper
from nds.modeling.models.nsg2.core import NsgWarpper2
from nds.modeling.models.nsg_sdf.core import NsgSDFWarpper
from nds.modeling.models.slot_attention.core import SlotAttentionAutoEncoder
from nds.modeling.models.star.core import STaR
from nds.modeling.models.starmo.starmo import STaRMo
from nds.modeling.models.star_ps.core import STaRPs
from nds.modeling.models.savi.savi import Savi
# from nds.modeling.models.nds.nds import NdsWarpper
from nds.modeling.models.star_sdf.star_sdf import STaRSdf
from nds.modeling.models.star_sdf.starsdfmo import STaRSdfMo
from nds.modeling.models.star_sdf.starsdfmouni import STaRSdfMoUni
from nds.modeling.models.star_mm.starmm import STaRMM
from nds.modeling.models.volsdf.volsdf import VolSDFWarpper
from nds.modeling.models.neural_diff.neuraldiff import NeuralDiffWarpper
from nds.modeling.models.barf.barf import BaRF
from nds.modeling.models.volsdf_surface.surface import ImplicitSurface
from nds.modeling.models.fbs.fbs import FlowBasedSegm

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
