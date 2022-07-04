import os
from yacs.config import CfgNode as CN

_C = CN()

_C.model = CN()
_C.model.device = "cuda"
_C.dbg = False
_C.deterministic = False
_C.evaltime = False

_C.model.meta_architecture = "resnet"

_C.model.yolact = CN()
_C.model.yolact.freeze_bn = True
_C.model.yolact.backbone = CN()
_C.model.yolact.backbone.args = [3, 4, 6, 3]
_C.model.yolact.backbone.selected_layers = [1, 2, 3]
_C.model.yolact.backbone.pred_aspect_ratios = [[[1, 0.5, 2]], [[1, 0.5, 2]], [[1, 0.5, 2]], [[1, 0.5, 2]],
                                               [[1, 0.5, 2]]]
_C.model.yolact.backbone.pred_scales = [[24], [48], [96], [192], [384]]
_C.model.yolact.backbone.preapply_sqrt = False
_C.model.yolact.backbone.use_pixel_scales = True
_C.model.yolact.backbone.use_square_anchors = True
_C.model.yolact.pretrained_backbone = '/raid/linghao/project_data/yolact/weights/resnet50-19c8e357.pth'

_C.model.yolact.mask_type = 1
_C.model.yolact.mask_proto_use_grid = False
_C.model.yolact.mask_proto_src = 0
_C.model.yolact.mask_proto_bias = False
_C.model.yolact.mask_proto_net = [(256, 3, {'padding': 1}), (256, 3, {'padding': 1}), (256, 3, {'padding': 1}),
                                  (None, -2, {}), (256, 3, {'padding': 1}), (32, 1, {})]
_C.model.yolact.use_maskiou = False
_C.model.yolact.share_prediction_module = True
_C.model.yolact.use_class_existence_loss = False
_C.model.yolact.use_semantic_segmentation_loss = True
_C.model.yolact.num_classes = 81
_C.model.yolact.mask_proto_split_prototypes_by_head = False
_C.model.yolact.mask_proto_prototypes_as_features = False
_C.model.yolact.extra_head_net = [(256, 3, {'padding': 1})]
_C.model.yolact.use_prediction_module = False
_C.model.yolact.head_layer_params = CN()
_C.model.yolact.head_layer_params.kernel_size = 3
_C.model.yolact.head_layer_params.padding = 1
_C.model.yolact.use_mask_scoring = False
_C.model.yolact.use_instance_coeff = False
_C.model.yolact.num_instance_coeffs = 64
_C.model.yolact.extra_layers = [0, 0, 0]
_C.model.yolact.mask_proto_coeff_gate = False
_C.model.yolact.nms_top_k = 200
_C.model.yolact.nms_conf_thresh = 0.05
_C.model.yolact.nms_thresh = 0.5
_C.model.yolact.positive_iou_threshold = 0.5
_C.model.yolact.negative_iou_threshold = 0.4
_C.model.yolact.ohem_negpos_ratio = 3
_C.model.yolact.use_class_balanced_conf = False
_C.model.yolact.eval_mask_branch = True
_C.model.yolact.mask_proto_prototypes_as_features_no_grad = False
_C.model.yolact.extra_head_net = [(256, 3, {'padding': 1})]
_C.model.yolact.use_yolo_regressors = False
_C.model.yolact.max_size = 550
_C.model.yolact.use_prediction_matching = False
_C.model.yolact.crowd_iou_threshold = 0.7
_C.model.yolact.use_change_matching = False
_C.model.yolact.train_boxes = True
_C.model.yolact.train_masks = True
_C.model.yolact.bbox_alpha = 1.5
_C.model.yolact.mask_alpha = 6.125
# _C.model.yolact.mask_proto_loss = None
_C.model.yolact.use_focal_loss = False
_C.model.yolact.use_sigmoid_focal_loss = False
_C.model.yolact.use_objectness_score = False
_C.model.yolact.mask_proto_crop = True
_C.model.yolact.mask_proto_normalize_emulate_roi_pooling = True
_C.model.yolact.mask_proto_remove_empty_masks = False
_C.model.yolact.mask_proto_binarize_downsampled_gt = True
_C.model.yolact.mask_proto_remove_empty_masks = False
_C.model.yolact.mask_proto_reweight_mask_loss = False
_C.model.yolact.mask_proto_reweight_coeff = 1
_C.model.yolact.mask_proto_crop_with_pred_box = False
_C.model.yolact.mask_proto_coeff_diversity_loss = False
_C.model.yolact.masks_to_train = 100
_C.model.yolact.mask_proto_double_loss = False
_C.model.yolact.mask_proto_normalize_mask_loss_by_sqrt_area = False
_C.model.yolact.mask_proto_normalize_emulate_roi_pooling = True
_C.model.yolact.ohem_use_most_confident = False
_C.model.yolact.use_class_balanced_conf = False
_C.model.yolact.conf_alpha = 1
_C.model.yolact.semantic_segmentation_alpha = 1
_C.model.yolact.ohem_use_most_confident = False
_C.model.yolact.max_num_detections = 100
_C.model.yolact.display_text = True
_C.model.yolact.display_scores = True
_C.model.yolact.score_threshold = 0.0
_C.model.yolact.top_k = 5
_C.model.yolact.class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                               'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                               'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                               'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                               'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                               'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                               'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                               'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                               'scissors', 'teddy bear', 'hair drier', 'toothbrush']
_C.model.yolact.nvis = 100

_C.model.yolact.fpn = CN()
_C.model.yolact.fpn.interpolation_mode = 'bilinear'
_C.model.yolact.fpn.num_downsample = 2
_C.model.yolact.fpn.num_features = 256
_C.model.yolact.fpn.pad = True
_C.model.yolact.fpn.relu_downsample_layers = False
_C.model.yolact.fpn.relu_pred_layers = True
_C.model.yolact.fpn.use_conv_downsample = True

_C.model.yolact_tracking = CN()
_C.model.yolact_tracking.pretrained_yolact = ''
_C.model.yolact_tracking.fix_yolact = True
_C.model.yolact_tracking.alpha = 0.1
_C.model.yolact_tracking.beta = 0.1
_C.model.yolact_tracking.thresh = 0.0
_C.model.yolact_tracking.track_head = CN()
_C.model.yolact_tracking.track_head.in_channels = 256
_C.model.yolact_tracking.track_head.roi_feat_size = 7
_C.model.yolact_tracking.track_head.match_coeff = [1.0, 2.0, 10]
_C.model.yolact_tracking.track_head.bbox_dummy_iou = 0
_C.model.yolact_tracking.track_head.num_fcs = 2
_C.model.yolact_tracking.track_head.fc_out_channels = 1024
_C.model.yolact_tracking.track_head.dynamic = True

_C.model.idispnet = CN()
_C.model.idispnet.maxdisp = 48
_C.model.idispnet.mindisp = -48
_C.model.idispnet.input_size = 224
_C.model.idispnet.pretrained_model = 'models/PSMNet/pretrained_model_KITTI2015.tar'

_C.model.idispnet.preprocess = CN()
_C.model.idispnet.preprocess.output_dir = ''
_C.model.idispnet.preprocess.prediction_template = ''
_C.model.idispnet.preprocess.masker_thresh = 0.5
_C.model.idispnet.preprocess.size = 224
_C.model.idispnet.preprocess.resize_mode = 'inverse_bilinear'

_C.model.drcnn = CN()
_C.model.drcnn.yolact_on = True
_C.model.drcnn.pretrained_yolact = ''
_C.model.drcnn.fix_yolact = True
_C.model.drcnn.ssim_coef = 0.0
_C.model.drcnn.ssim_intercept = 0.0
_C.model.drcnn.ssim_std = 0.0
_C.model.drcnn.idispnet_on = False
_C.model.drcnn.detector_3d_on = False
_C.model.drcnn.retvalid = True

_C.dataset = CN()
_C.dataset.kitti = CN()
_C.dataset.kitti_object = CN()
_C.dataset.kitti_object.filter_empty = False
_C.dataset.kitti_object.offline_2d_predictions_path = ''
_C.dataset.kitti_object.shape_prior_base = 'vob'
_C.dataset.kitti_object.remove_ignore = True
_C.dataset.kitti_object.use_gray = False

_C.dataset.kittiroi = CN()
_C.dataset.kittiroi.root = 'data/???'
_C.dataset.kittiroi.maxdisp = 48
_C.dataset.kittiroi.mindisp = -48

_C.dataset.kitti_tracking = CN()
_C.dataset.kitti_tracking.use_gray = False

_C.input = CN()
_C.input.transforms = []
_C.input.transforms_test = []
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
