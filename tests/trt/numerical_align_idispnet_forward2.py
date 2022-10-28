import os

import torch

from disprcnn.engine.defaults import default_argument_parser
from disprcnn.config import cfg
from disprcnn.trainer.build import build_trainer
from disprcnn.utils.comm import get_rank
from disprcnn.utils.logger import setup_logger

parser = default_argument_parser()
args = parser.parse_args()
args.config_file = "configs/drcnn/kitti_tracking/pointpillars_112_600x300_demo.yaml"

cfg.merge_from_file(args.config_file)
cfg.merge_from_list(args.opts)
if cfg.output_dir == '':
    assert args.config_file.startswith('configs') and args.config_file.endswith('.yaml')
    cfg.output_dir = args.config_file[:-5].replace('configs', 'models')
if 'PYCHARM_HOSTED' in os.environ:
    cfg.dataloader.num_workers = 0
cfg.mode = 'test'
cfg.freeze()
logger = setup_logger(cfg.output_dir, get_rank(), 'logtest.txt')
trainer = build_trainer(cfg)
trainer.resume()

valid_ds = trainer.valid_dl.dataset
data0 = valid_ds[0]
calib = data0['targets']['left'].extra_fields['calib']
model = trainer.model
model = model.idispnet
model.eval()
model.cuda()

left_roi_images, right_roi_images = torch.load('tmp/left_right_roi_images.pth')
outputs = model.forward_onnx(left_roi_images[0:1], right_roi_images[0:1])
print()
