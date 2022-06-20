import os
from yacs.config import CfgNode as CN

_C = CN()

_C.model = CN()
_C.model.device = "cuda"
_C.model.dilate_num = 26  # 26 or 2. 2 is wrong, only for speed test
# _C.model.mode = "train"  # or test. NOTE THAT THIS IS NOT THE SAME AS MODEL.TRAINING().
_C.dbg = False
_C.deterministic = False
_C.evaltime = False
_C.model.meta_architecture = "NeRF"
_C.model.resnet = CN()
_C.model.resnet.num_classes = 1000
_C.model.resnet.pretrained = True

_C.dataset = CN()
_C.dataset.kitti = CN()

_C.input = CN()
_C.input.transforms = []
_C.input.shuffle = True

_C.datasets = CN()
_C.datasets.train = ()
_C.datasets.test = ""

_C.dataloader = CN()
_C.dataloader.num_workers = 0
_C.dataloader.collator = 'DefaultBatchCollator'
_C.dataloader.pin_memory = False

_C.solver = CN()
_C.solver.num_epochs = 1
_C.solver.max_lr = 0.001
_C.solver.end_lr = 0.0001
_C.solver.end_pose_lr = 0.00001
_C.solver.bias_lr_factor = 1
_C.solver.momentum = 0.9
_C.solver.weight_decay = 0.0005
_C.solver.weight_decay_bias = 0.0
_C.solver.gamma = 0.1
_C.solver.gamma_pose = 0.1
_C.solver.lrate_decay = 250
_C.solver.lrate_decay_pose = 250
_C.solver.steps = (30000,)
_C.solver.warmup_factor = 1.0 / 3
_C.solver.warmup_iters = 500
_C.solver.warmup_method = "linear"
_C.solver.num_iters = 10000  # volsdf
_C.solver.min_factor = 0.1  # volsdf
_C.solver.pose_lr = 0.00005  # volsdf

_C.solver.optimizer = 'Adam'
_C.solver.scheduler = 'OneCycleScheduler'
_C.solver.scheduler_decay_thresh = 0.00005
_C.solver.do_grad_clip = False
_C.solver.grad_clip_type = 'norm'  # norm or value
_C.solver.grad_clip = 1.0
_C.solver.ds_len = -1
_C.solver.batch_size = 1
_C.solver.loss_function = ''
####save ckpt configs#####
_C.solver.save_min_loss = 20.0
_C.solver.save_every = False
_C.solver.save_freq = 1
_C.solver.save_mode = 'epoch'  # epoch or iteration
_C.solver.val_freq = 1
_C.solver.save_last_only = False
_C.solver.empty_cache = True
# _C.solver.force_no_resume = False
# save model config:
#  --->save model when smaller val loss is detected.
# save_every: True, save_mode: epoch --->save model when epoch % save_freq==0
# save_every: True, save_mode: iteration --->save model when epoch % save_freq==0
_C.solver.trainer = "base"
_C.solver.load_model = ""
_C.solver.load = ""
_C.solver.print_it = False
_C.solver.detect_anomaly = False
_C.solver.convert_sync_batchnorm = False
_C.solver.ddp_version = 'torch'  # or nds
_C.solver.broadcast_buffers = False
_C.solver.find_unused_parameters = False
_C.solver.resume = False
_C.solver.dist = CN()
_C.solver.dist.sampler = CN()
_C.solver.dist.sampler.type = 'pytorch'
_C.solver.dist.sampler.shuffle = False

_C.test = CN()
_C.test.batch_size = 1
_C.test.evaluators = []
_C.test.visualizer = 'default'
_C.test.force_recompute = True
_C.test.do_evaluation = True
_C.test.do_visualization = False
_C.test.eval_all = False
_C.test.eval_all_min = 0
_C.test.save_predictions = False
_C.test.training_mode = False
_C.test.ckpt_dir = ''

_C.output_dir = ''
_C.backup_src = True
_C.mode = 'train'

_C.paths_catalog = os.path.join(os.path.dirname(__file__), "paths_catalog.py")
