import os.path as osp
import open3d as o3d
import copy
import glob

import numpy as np
import trimesh
import zarr
import torch
from torch.utils.data import Dataset
from tqdm import trange
from easydict import EasyDict as edict

# from datasets.dataloader import get_dataloader
from nds.data.datasets.indoor import IndoorDataset, get_dataloader
# from nds.modeling.models.predator.lib.benchmark_utils import ransac_pose_estimation
from nds.modeling.models.predator.lib.benchmark_utils import to_o3d_pcd, to_o3d_feats
from nds.modeling.models.predator.lib.utils import load_obj, setup_seed, load_config

from nds.utils.cam_utils import cam_fufvcucv_to_matrix
from nds.utils.pn_utils import to_array, random_choice
from nds.utils.tsdf_fusion_python import TSDFVolume
from nds.utils.utils_3d import transform_points, depth_to_rect
from nds.utils.vis3d_ext import Vis3D

from nds.modeling.models.predator.lib.utils import load_config
from nds.modeling.models.predator.architectures import KPFCNN


def ransac_pose_estimation(src_pcd, tgt_pcd, src_feat, tgt_feat, mutual=False, distance_threshold=0.05, ransac_n=3):
    """
    RANSAC pose estimation with two checkers
    We follow D3Feat to set ransac_n = 3 for 3DMatch and ransac_n = 4 for KITTI.
    For 3DMatch dataset, we observe significant improvement after changing ransac_n from 4 to 3.
    """
    if (mutual):
        if (torch.cuda.device_count() >= 1):
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        src_feat, tgt_feat = to_tensor(src_feat), to_tensor(tgt_feat)
        scores = torch.matmul(src_feat.to(device), tgt_feat.transpose(0, 1).to(device)).cpu()
        selection = mutual_selection(scores[None, :, :])[0]
        row_sel, col_sel = np.where(selection)
        corrs = o3d.utility.Vector2iVector(np.array([row_sel, col_sel]).T)
        src_pcd = to_o3d_pcd(src_pcd)
        tgt_pcd = to_o3d_pcd(tgt_pcd)
        result_ransac = o3d.registration.registration_ransac_based_on_correspondence(
            source=src_pcd, target=tgt_pcd, corres=corrs,
            max_correspondence_distance=distance_threshold,
            estimation_method=o3d.registration.TransformationEstimationPointToPoint(False),
            ransac_n=4,
            criteria=o3d.registration.RANSACConvergenceCriteria(50000, 1000))
    else:
        src_pcd = to_o3d_pcd(src_pcd)
        tgt_pcd = to_o3d_pcd(tgt_pcd)
        src_feats = to_o3d_feats(src_feat)
        tgt_feats = to_o3d_feats(tgt_feat)

        result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            src_pcd, tgt_pcd, src_feats, tgt_feats, False, distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False), ransac_n,
            [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
             o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
            o3d.pipelines.registration.RANSACConvergenceCriteria(50000, 1000))

    return result_ransac


class ThreeDMatchDemo(Dataset):
    """
    Load subsampled coordinates, relative rotation and translation
    Output(torch.Tensor):
        src_pcd:        [N,3]
        tgt_pcd:        [M,3]
        rot:            [3,3]
        trans:          [3,1]
    """

    def __init__(self, config, src_pcd, tgt_pcd):
        super(ThreeDMatchDemo, self).__init__()
        self.config = config
        self.src_pcd = src_pcd
        self.tgt_pcd = tgt_pcd

    def __len__(self):
        return 1

    def __getitem__(self, item):
        src_pcd = self.src_pcd.astype(np.float32)
        tgt_pcd = self.tgt_pcd.astype(np.float32)

        src_feats = np.ones_like(src_pcd[:, :1]).astype(np.float32)
        tgt_feats = np.ones_like(tgt_pcd[:, :1]).astype(np.float32)

        # fake the ground truth information
        rot = np.eye(3).astype(np.float32)
        trans = np.ones((3, 1)).astype(np.float32)
        correspondences = torch.ones(1, 2).long()

        return src_pcd, tgt_pcd, src_feats, tgt_feats, rot, trans, correspondences, src_pcd, tgt_pcd, torch.ones(1)


class PreCKPT:
    config = None
    neighborhood_limits = None

    @staticmethod
    def get_instance():
        if PreCKPT.config is None:
            predator_root = osp.expanduser('~/PycharmProjects/OverlapPredator')
            config = load_config(osp.join(predator_root, 'configs/test/indoor.yaml'))
            config = edict(config)
            config.pretrain = osp.join(predator_root, config.pretrain)
            config.train_info = osp.join(predator_root, config.train_info)
            config.root = osp.join(predator_root, config.root)
            if config.gpu_mode:
                config.device = torch.device('cuda')
            else:
                config.device = torch.device('cpu')

            # model initialization
            config.architecture = [
                'simple',
                'resnetb',
            ]
            for i in range(config.num_layers - 1):
                config.architecture.append('resnetb_strided')
                config.architecture.append('resnetb')
                config.architecture.append('resnetb')
            for i in range(config.num_layers - 2):
                config.architecture.append('nearest_upsample')
                config.architecture.append('unary')
            config.architecture.append('nearest_upsample')
            config.architecture.append('last_unary')
            config.model = KPFCNN(config).to(config.device)

            # create dataset and dataloader
            info_train = load_obj(config.train_info)
            train_set = IndoorDataset(info_train, config, data_augmentation=True)

            _, neighborhood_limits = get_dataloader(dataset=train_set,
                                                    batch_size=config.batch_size,
                                                    shuffle=True,
                                                    num_workers=config.num_workers,
                                                    )

            # load pretrained weights
            assert config.pretrain != None
            state = torch.load(config.pretrain)
            config.model.load_state_dict(state['state_dict'])
            PreCKPT.config = config
            PreCKPT.neighborhood_limits = neighborhood_limits
        return PreCKPT.config, PreCKPT.neighborhood_limits


def predator_api(src_pc, tgt_pc):
    if isinstance(src_pc, trimesh.Trimesh):
        src_pc = src_pc.vertices
    if isinstance(src_pc, torch.Tensor):
        src_pc = src_pc.cpu().numpy()
    if isinstance(tgt_pc, trimesh.Trimesh):
        tgt_pc = tgt_pc.vertices
    if isinstance(tgt_pc, torch.Tensor):
        tgt_pc = tgt_pc.cpu().numpy()
    # if normalize_scale:
    #     scaling = 1.0 / (src_pc.max(0) - src_pc.min(0)).max()
    #     src_pc = src_pc * scaling
    # scaling = 1.0 / (tgt_pc.max(0) - tgt_pc.min(0)).max()
    # tgt_pc = tgt_pc * scaling
    # if normalize_position:
    #     src_pc = src_pc - src_pc.min(0)
    #     tgt_pc = tgt_pc - tgt_pc.min(0)
    config, neighborhood_limits = PreCKPT.get_instance()
    demo_set = ThreeDMatchDemo(config, src_pc, tgt_pc)
    demo_loader, _ = get_dataloader(dataset=demo_set,
                                    batch_size=config.batch_size,
                                    shuffle=False,
                                    num_workers=0,
                                    neighborhood_limits=neighborhood_limits)
    # do pose estimation
    config.model.eval()
    c_loader_iter = demo_loader.__iter__()
    with torch.no_grad():
        inputs = c_loader_iter.next()
        ##################################
        # load inputs to device.
        for k, v in inputs.items():
            if type(v) == list:
                inputs[k] = [item.to(config.device) for item in v]
            else:
                inputs[k] = v.to(config.device)

        ###############################################
        # forward pass
        feats, scores_overlap, scores_saliency = config.model(inputs)  # [N1, C1], [N2, C2]
        pcd = inputs['points'][0]
        len_src = inputs['stack_lengths'][0][0]
        c_rot, c_trans = inputs['rot'], inputs['trans']
        correspondence = inputs['correspondences']

        src_pcd, tgt_pcd = pcd[:len_src], pcd[len_src:]
        src_raw = copy.deepcopy(src_pcd)
        tgt_raw = copy.deepcopy(tgt_pcd)
        src_feats, tgt_feats = feats[:len_src].detach().cpu(), feats[len_src:].detach().cpu()
        src_overlap, src_saliency = scores_overlap[:len_src].detach().cpu(), scores_saliency[:len_src].detach().cpu()
        tgt_overlap, tgt_saliency = scores_overlap[len_src:].detach().cpu(), scores_saliency[len_src:].detach().cpu()

        ########################################
        # do probabilistic sampling guided by the score
        src_scores = src_overlap * src_saliency
        tgt_scores = tgt_overlap * tgt_saliency

        if (src_pcd.size(0) > config.n_points):
            idx = np.arange(src_pcd.size(0))
            probs = (src_scores / src_scores.sum()).numpy().flatten()
            idx = np.random.choice(idx, size=config.n_points, replace=False, p=probs)
            src_pcd, src_feats = src_pcd[idx], src_feats[idx]
        if (tgt_pcd.size(0) > config.n_points):
            idx = np.arange(tgt_pcd.size(0))
            probs = (tgt_scores / tgt_scores.sum()).numpy().flatten()
            idx = np.random.choice(idx, size=config.n_points, replace=False, p=probs)
            tgt_pcd, tgt_feats = tgt_pcd[idx], tgt_feats[idx]

        ########################################
        # run ransac and draw registration
        tsfm = ransac_pose_estimation(src_pcd, tgt_pcd, src_feats, tgt_feats, mutual=False)
        return tsfm
        # draw_registration_result(src_raw, tgt_raw, src_overlap, tgt_overlap, src_saliency, tgt_saliency, tsfm)


def do_predator(pts0, pts1, vis3d, i, init=np.eye(4)):
    pcd0 = pts0.astype(np.float32)
    pcd1 = pts1.astype(np.float32)
    pcd0, _ = random_choice(pcd0, 10000, dim=0, replace=True)
    pcd1, _ = random_choice(pcd1, 10000, dim=0, replace=True)
    vis3d.add_point_cloud(pcd0, name=f'pcd0_{i}')
    vis3d.add_point_cloud(pcd1, name=f'pcd1_{i}')
    obj_pose = predator_api(pcd0, pcd1)
    vis3d.add_point_cloud(transform_points(pcd0, init), name=f'tpcdinit0_{i}')
    vis3d.add_point_cloud(transform_points(pcd0, obj_pose), name=f'tpcd0_{i}')
    return obj_pose


def main():
    vis3d = Vis3D(
        xyz_pattern=('x', 'y', 'z'),
        out_folder="dbg",
        sequence="icp_dtu_meshes",
        # auto_increase=,
        # enable=,
    )
    mesh0 = trimesh.load_mesh('/mnt/data/Datasets/DTU/MVS Data/Surfaces/tola/tola055_l3_surf_11_trim_8.ply')
    mesh1 = trimesh.load_mesh('/mnt/data/Datasets/DTU/MVS Data/Surfaces/tola/tola056_l3_surf_11_trim_8.ply')
    do_predator(mesh0.vertices, mesh1.vertices, vis3d, 0)


if __name__ == '__main__':
    main()
