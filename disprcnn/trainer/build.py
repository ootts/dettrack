from nds.registry import TRAINERS
from nds.trainer.barf import BaRFTrainer
from nds.trainer.base import BaseTrainer
from nds.trainer.depthfusion import DepthFusionTrainer
from nds.trainer.depthfusion_onetime import DepthFusionOTTrainer
from nds.trainer.nerf import NeRFTrainer
from nds.trainer.nsg_trainer import NsgTrainer
from nds.trainer.slot_attn import SlotAttnTrainer
from nds.trainer.star import STaRTrainer
from nds.trainer.savi import SaviTrainer
from nds.trainer.starmo import STaRMoTrainer
from nds.trainer.starsdf import STaRSdfTrainer
from nds.trainer.starsdfmouni import STaRSdfMoUniTrainer
from nds.trainer.volsdf import VolSDFTrainer
from nds.trainer.surface import SurfaceTrainer


@TRAINERS.register('base')
def build_base_trainer(cfg):
    return BaseTrainer(cfg)


@TRAINERS.register('nerf')
def build(cfg):
    return NeRFTrainer(cfg)


@TRAINERS.register('barf')
def build(cfg):
    return BaRFTrainer(cfg)


@TRAINERS.register('slot_attn')
def build(cfg):
    return SlotAttnTrainer(cfg)


@TRAINERS.register('savi')
def build(cfg):
    return SaviTrainer(cfg)


@TRAINERS.register('nsg')
def build(cfg):
    return NsgTrainer(cfg)


@TRAINERS.register('star')
def build(cfg):
    return STaRTrainer(cfg)


@TRAINERS.register('starsdf')
def build(cfg):
    return STaRSdfTrainer(cfg)


@TRAINERS.register('starsdfmouni')
def build(cfg):
    return STaRSdfMoUniTrainer(cfg)


@TRAINERS.register('starmo')
def build(cfg):
    return STaRMoTrainer(cfg)


@TRAINERS.register('volsdf')
def build(cfg):
    return VolSDFTrainer(cfg)


@TRAINERS.register('depthfusion')
def build(cfg):
    return DepthFusionTrainer(cfg)


@TRAINERS.register('depthfusionot')
def build(cfg):
    return DepthFusionOTTrainer(cfg)


@TRAINERS.register('surface')
def build(cfg):
    return SurfaceTrainer(cfg)


def build_trainer(cfg) -> BaseTrainer:
    return TRAINERS[cfg.solver.trainer](cfg)
