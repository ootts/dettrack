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
_C.model.yolact.backbone.args = ([3, 4, 6, 3],)
_C.model.yolact.backbone.selected_layers = [1, 2, 3]
_C.model.yolact.backbone.pred_aspect_ratios = [[[1, 0.5, 2]], [[1, 0.5, 2]], [[1, 0.5, 2]], [[1, 0.5, 2]],
                                               [[1, 0.5, 2]]]
_C.model.yolact.backbone.pred_scales = [[24], [48], [96], [192], [384]]
_C.model.yolact.backbone.preapply_sqrt = False
_C.model.yolact.backbone.use_pixel_scales = True
_C.model.yolact.backbone.use_square_anchors = True
_C.model.yolact.pretrained_backbone = '../yolact/weights/resnet50-19c8e357.pth'

_C.model.yolact.mask_type = 1
_C.model.yolact.mask_proto_use_grid = False
_C.model.yolact.mask_proto_src = 0
_C.model.yolact.mask_proto_bias = False
_C.model.yolact.mask_proto_net = [(256, 3, {'padding': 1}), (256, 3, {'padding': 1}), (256, 3, {'padding': 1}),
                                  (None, -2, {}), (256, 3, {'padding': 1}), (32, 1, {})]
_C.model.yolact.use_maskiou = False
_C.model.yolact.maskiou_alpha = 25
_C.model.yolact.rescore_bbox = False
_C.model.yolact.rescore_mask = False
_C.model.yolact.discard_mask_area = 25
_C.model.yolact.maskious_to_train = -1

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

_C.model.pointpillars = CN()
_C.model.pointpillars.pretrained_model = ""
_C.model.pointpillars.preprocess = CN()
_C.model.pointpillars.preprocess.save_dir = ""
_C.model.pointpillars.preprocess.velodyne_template = ""
_C.model.pointpillars.preprocess.velodyne_reduced_dir = ""

_C.model.pointpillars.num_classes = 1
_C.model.pointpillars.vis_threshold = 0.5
# _C.model.pointpillars.use_rotate_nms = False
_C.model.pointpillars.multiclass_nms = False
_C.model.pointpillars.use_bev = False
_C.model.pointpillars.lidar_only = False
# _C.model.pointpillars.use_groupnorm = False
_C.model.pointpillars.pos_class_weight = 1.0
_C.model.pointpillars.neg_class_weight = 1.0
_C.model.pointpillars.voxel_feature_extractor = CN()
# _C.model.pointpillars.voxel_feature_extractor.module_class_name = "PillarFeatureNet"
_C.model.pointpillars.voxel_feature_extractor.num_filters = [64]
_C.model.pointpillars.voxel_feature_extractor.with_distance = False
_C.model.pointpillars.middle_feature_extractor = CN()
# _C.model.pointpillars.middle_feature_extractor.module_class_name = "PointPillarsScatter"
_C.model.pointpillars.middle_feature_extractor.module_class_name = "PointPillarsScatter"
_C.model.pointpillars.rpn = CN()
_C.model.pointpillars.rpn.module_class_name = "RPN"
_C.model.pointpillars.rpn.layer_nums = [3, 5, 5]
_C.model.pointpillars.rpn.layer_strides = [2, 2, 2]
_C.model.pointpillars.rpn.num_filters = [64, 128, 256]
_C.model.pointpillars.rpn.upsample_strides = [1, 2, 4]
_C.model.pointpillars.rpn.num_upsample_filters = [128, 128, 128]
# _C.model.pointpillars.rpn.num_groups = 32
_C.model.pointpillars.use_sigmoid_score = True
_C.model.pointpillars.loss = CN()
_C.model.pointpillars.loss.localization_loss = CN()
_C.model.pointpillars.loss.localization_loss.weighted_smooth_l1 = CN()
_C.model.pointpillars.loss.localization_loss.weighted_smooth_l1.sigma = 3.0
_C.model.pointpillars.loss.localization_loss.weighted_smooth_l1.code_weight = [1.0, 1.0, 1, 1, 1, 1]

_C.model.pointpillars.loss.classification_loss = CN()
_C.model.pointpillars.loss.classification_loss.weighted_sigmoid_focal = CN()
_C.model.pointpillars.loss.classification_loss.weighted_sigmoid_focal.anchorwise_output = True
_C.model.pointpillars.loss.classification_loss.weighted_sigmoid_focal.gamma = 2.0
_C.model.pointpillars.loss.classification_loss.weighted_sigmoid_focal.alpha = 0.25
_C.model.pointpillars.loss.classification_weight = 1.0
_C.model.pointpillars.loss.localization_weight = 2.0

_C.model.pointpillars.encode_rad_error_by_sin = True
_C.model.pointpillars.encode_background_as_zeros = True
_C.model.pointpillars.nms_pre_max_size = 1000
_C.model.pointpillars.nms_post_max_size = 300
_C.model.pointpillars.nms_score_threshold = 0.05000000074505806
_C.model.pointpillars.nms_iou_threshold = 0.5
_C.model.pointpillars.post_center_limit_range = [0.0, -39.68000030517578, -5.0, 69.12000274658203, 39.68000030517578,
                                                 5.0]
# _C.model.pointpillars.use_direction_classifier = True
_C.model.pointpillars.direction_loss_weight = 0.20000000298023224
_C.model.pointpillars.pos_class_weight = 1.0
_C.model.pointpillars.neg_class_weight = 1.0
_C.model.pointpillars.loss_norm_type = "NormByNumPositives"
_C.model.pointpillars.box_coder = "ground_box3d_coder"  # todo?

_C.model.pointpillars.target_assigner = CN()
_C.model.pointpillars.target_assigner.anchor_generators = CN()
_C.model.pointpillars.target_assigner.anchor_generators.sizes = [1.600000023841858,
                                                                 3.9000000953674316,
                                                                 1.559999942779541]
_C.model.pointpillars.target_assigner.anchor_generators.strides = [0.3199999928474426,
                                                                   0.3199999928474426, 0.0]
_C.model.pointpillars.target_assigner.anchor_generators.offsets = [0.1599999964237213,
                                                                   -39.52000045776367,
                                                                   -1.7799999713897705]
_C.model.pointpillars.target_assigner.anchor_generators.rotations = [0, 1.5700000524520874]
_C.model.pointpillars.target_assigner.anchor_generators.matched_threshold = 0.6000000238418579
_C.model.pointpillars.target_assigner.anchor_generators.unmatched_threshold = 0.44999998807907104
_C.model.pointpillars.target_assigner.anchor_generators.class_name = ""
_C.model.pointpillars.target_assigner.sample_positive_fraction = -1.0
_C.model.pointpillars.target_assigner.sample_size = 512
_C.model.pointpillars.target_assigner.region_similarity_calculator = "nearest_iou_similarity"

_C.model.pointpillars.num_point_features = 4

_C.model.pointpillars.localization_loss = CN()
# _C.model.pointpillars.localization_loss.type = "weighted_smooth_l1"
_C.model.pointpillars.localization_loss.sigma = 3.0
_C.model.pointpillars.localization_loss.code_weight = [1.0, 1, 1, 1, 1, 1, 1]

_C.model.pointpillars.classification_loss = CN()
# _C.model.pointpillars.classification_loss.type = "weighted_sigmoid_focal"
_C.model.pointpillars.classification_loss.anchorwise_output = True
_C.model.pointpillars.classification_loss.gamma = 2.0
_C.model.pointpillars.classification_loss.alpha = 0.25

_C.model.pointpillars.classification_weight = 1.0
_C.model.pointpillars.localization_weight = 2.0

_C.voxel_generator = CN()
_C.voxel_generator.voxel_size = [0.1599999964237213, 0.1599999964237213, 4.0]
_C.voxel_generator.point_cloud_range = [0.0, -39.68000030517578, -3.0, 69.12000274658203, 39.68000030517578, 1.0]
_C.voxel_generator.max_number_of_points_per_voxel = 100

_C.model.drcnn = CN()
_C.model.drcnn.yolact_on = True
_C.model.drcnn.pretrained_yolact = ''
_C.model.drcnn.yolact_tracking_on = False
_C.model.drcnn.pretrained_yolact_tracking = ''
_C.model.drcnn.fix_yolact = True
# _C.model.drcnn.ssim_coef = 0.0
# _C.model.drcnn.ssim_intercept = 0.0
# _C.model.drcnn.ssim_std = 0.0
_C.model.drcnn.ssim_coefs = []
_C.model.drcnn.ssim_intercepts = []
_C.model.drcnn.ssim_stds = []
_C.model.drcnn.mask_mode = 'poly'
_C.model.drcnn.idispnet_on = False
_C.model.drcnn.fix_idispnet = True
_C.model.drcnn.detector_3d_on = False
_C.model.drcnn.detector_3d = CN()
# _C.model.drcnn.detector_3d.use_lidar = False
_C.model.drcnn.detector_3d.shuffle_points = True
_C.model.drcnn.detector_3d.combine_2d3d = False
_C.model.drcnn.detector_3d.feature_map_size = [1, 248, 216]
_C.model.drcnn.detector_3d.max_number_of_voxels = 12000
_C.model.drcnn.detector_3d.anchor_area_threshold = 1.0
_C.model.drcnn.detector_3d.aug_on = False
_C.model.drcnn.detector_3d.aug = CN()
_C.model.drcnn.detector_3d.aug.global_rotation_noise = [-0.7853981852531433, 0.7853981852531433]
_C.model.drcnn.detector_3d.aug.global_scaling_noise = [0.949999988079071, 1.0499999523162842]
_C.model.drcnn.retvalid = True
_C.model.drcnn.nvis = 10

_C.dataset = CN()

_C.dataset.coco = CN()
_C.dataset.coco.use_gray = False
_C.dataset.coco.class_only = ""

_C.dataset.kitti_kins = CN()
_C.dataset.kitti_kins.use_gray = False
_C.dataset.kitti_kins.classes = ("__background__", "car", "dontcare")
_C.dataset.kitti_kins.remove_empty = False

_C.dataset.kitti_object = CN()
_C.dataset.kitti_object.filter_empty = False
_C.dataset.kitti_object.offline_2d_predictions_path = ''
_C.dataset.kitti_object.shape_prior_base = 'vob'
_C.dataset.kitti_object.remove_ignore = True
_C.dataset.kitti_object.use_gray = False
_C.dataset.kitti_object.load_lidar = False
_C.dataset.kitti_object.classes = ("__background__", "car", "dontcare")

_C.dataset.kittiroi = CN()
_C.dataset.kittiroi.root = 'data/???'
_C.dataset.kittiroi.maxdisp = 48
_C.dataset.kittiroi.mindisp = -48

_C.dataset.kitti_tracking = CN()
_C.dataset.kitti_tracking.use_gray = False
_C.dataset.kitti_tracking.classes = ("__background__", "car", "dontcare")

_C.dataset.kitti_tracking_stereo = CN()
_C.dataset.kitti_tracking_stereo.use_gray = False

_C.dataset.kitti_velodyne = CN()
_C.dataset.kitti_velodyne.without_reflectivity = False
# _C.dataset.kitti_velodyne.out_size_factor = 2
# _C.dataset.kitti_velodyne.generate_bev = False
_C.dataset.kitti_velodyne.info_path = "data/kitti_second/kitti_infos_%s.pkl"
_C.dataset.kitti_velodyne.root_path = 'data/kitti_second'
_C.dataset.kitti_velodyne.num_point_features = 4
_C.dataset.kitti_velodyne.feature_map_size = [1, 248, 216]
_C.dataset.kitti_velodyne.db_sample = True
_C.dataset.kitti_velodyne.db_sampler = CN()
_C.dataset.kitti_velodyne.db_sampler.database_info_path = "data/kitti_second/kitti_dbinfos_train.pkl"
_C.dataset.kitti_velodyne.db_sampler.global_random_rotation_range_per_object = [0.0, 0.0]
_C.dataset.kitti_velodyne.db_sampler.rate = 1.0
_C.dataset.kitti_velodyne.db_sampler.sample_groups = []
# _C.dataset.kitti_velodyne.db_sampler.sample_groups.name_to_max_num = CN()
# _C.dataset.kitti_velodyne.db_sampler.sample_groups.key = "Car"
# _C.dataset.kitti_velodyne.db_sampler.sample_groups.value = 15
_C.dataset.kitti_velodyne.db_sampler.database_prep_steps = []
_C.dataset.kitti_velodyne.class_names = ['Car']
_C.dataset.kitti_velodyne.max_number_of_voxels = 12000
_C.dataset.kitti_velodyne.remove_unknown_examples = False
_C.dataset.kitti_velodyne.shuffle_points = True
_C.dataset.kitti_velodyne.groundtruth_rotation_uniform_noise = [-0.15707963705062866, 0.15707963705062866]
_C.dataset.kitti_velodyne.groundtruth_localization_noise_std = [0.25, 0.25, 0.25]
_C.dataset.kitti_velodyne.global_rotation_uniform_noise = [-0.7853981852531433, 0.7853981852531433]
_C.dataset.kitti_velodyne.global_scaling_uniform_noise = [0.949999988079071, 1.0499999523162842]
_C.dataset.kitti_velodyne.global_random_rotation_range_per_object = [0.0, 0.0]
_C.dataset.kitti_velodyne.anchor_area_threshold = 1.0
# _C.dataset.kitti_velodyne.groundtruth_points_drop_percentage = 0.0
# _C.dataset.kitti_velodyne.groundtruth_drop_max_keep_points = 15
_C.dataset.kitti_velodyne.remove_points_after_sample = False
# _C.dataset.kitti_velodyne.remove_environment = False
# _C.dataset.kitti_velodyne.use_group_id = False

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
_C.solver.tb_freq = 1

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

_C.solver.strict = True

_C.test = CN()
_C.test.batch_size = 1
_C.test.evaluators = []
_C.test.visualizer = 'default'
_C.test.force_recompute = True
_C.test.do_evaluation = True
_C.test.do_visualization = False
_C.test.eval_all = False
_C.test.eval_all_min = 0
_C.test.eval_all_attr = "solver.load_model"
_C.test.save_predictions = False
_C.test.training_mode = False
_C.test.compress_results = False
_C.test.ckpt_dir = ''

_C.output_dir = ''
_C.backup_src = True
_C.mode = 'train'

_C.paths_catalog = os.path.join(os.path.dirname(__file__), "paths_catalog.py")
