from disprcnn.modeling.models.yolact.submodules import *
from .yolact import Yolact


class YolactTracking(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.total_cfg = cfg
        self.cfg = cfg.model.yolact_tracking
        self.yolact = Yolact(cfg)
        ckpt = torch.load(self.cfg.pretrained_yolact, 'cpu')
        self.yolact.load_state_dict(ckpt['model'])
        self.track_head = None  # todo: build track_head
        print()

    def forward(self, dps):
        # todo: 1. forward yolact for each frame
        # todo: 2. roi align features
        # todo: 3. forward track_head
        # todo: 4. make gt and compute loss
        print()

    def train(self, mode=True):
        super(YolactTracking, self).train(mode)
        self.yolact.train(mode=False)
