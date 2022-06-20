import warnings
from itertools import product

import numpy as np
import torch
import torch.nn.functional as F
import torchsparse as ts
import torchsparse.nn as spnn
import torchsparse.nn.functional as spf
import trimesh
from packaging import version
from skimage.measure import marching_cubes

if version.parse(ts.__version__) < version.parse("1.4.0"):
    from torchsparse.sparse_tensor import SparseTensor
    from torchsparse.point_tensor import PointTensor
    from torchsparse.utils.helpers import *
else:
    from torchsparse.tensor import SparseTensor, PointTensor

from dl_ext.timer import EvalTime
from disprcnn.utils.pn_utils import norm, to_array, clone, intersect1d_pytorch


# x: SparseTensor, z: PointTensor
# return: PointTensor
def voxel_to_point(x, z, nearest=False):
    if z.shape[0] == 0:
        return z
    assert x.C.shape[1] == 4
    assert z.C.shape[1] == 4
    if z.idx_query is None or z.weights is None or z.idx_query.get(
            x.s) is None or z.weights.get(x.s) is None:
        kr = KernelRegion(2, x.s, 1)
        off = kr.get_kernel_offset().to(z.F.device)
        # old_hash = kernel_hash_gpu(torch.floor(z.C).int(), off)
        old_hash = spf.sphash(
            torch.cat([
                torch.floor(z.C[:, :3] / x.s).int() * x.s,
                z.C[:, -1].int().view(-1, 1)
            ], 1), off)
        pc_hash = spf.sphash(x.C.to(z.F.device))
        idx_query = spf.sphashquery(old_hash, pc_hash)
        if idx_query.ndim == 1:
            idx_query = idx_query.unsqueeze(1)
        weights = spf.calc_ti_weights(z.C, idx_query,
                                      scale=x.s).transpose(0, 1).contiguous()
        idx_query = idx_query.transpose(0, 1).contiguous()
        if nearest:
            weights[:, 1:] = 0.
            idx_query[:, 1:] = -1
        new_feat = spf.spdevoxelize(x.F, idx_query, weights)
        new_tensor = PointTensor(new_feat,
                                 z.C,
                                 idx_query=z.idx_query,
                                 weights=z.weights)
        new_tensor.additional_features = z.additional_features
        new_tensor.idx_query[x.s] = idx_query
        new_tensor.weights[x.s] = weights
        z.idx_query[x.s] = idx_query
        z.weights[x.s] = weights

    else:
        new_feat = spf.spdevoxelize(x.F, z.idx_query.get(x.s), z.weights.get(x.s))
        new_tensor = PointTensor(new_feat,
                                 z.C,
                                 idx_query=z.idx_query,
                                 weights=z.weights)
        new_tensor.additional_features = z.additional_features

    return new_tensor


def interpolate_feature(x_C, x_F, z_C, x_mask=None, z_mask=None):
    """

    :param x_C: B,N,3
    :param x_F: B,N,C
    :param z_C: B,M,3
    :param x_mask: B,N
    :param z_mask: B,M
    :return: z_F: B,M,C
    """
    B, N, _ = x_C.shape
    C = x_F.shape[-1]
    M = z_C.shape[1]
    if x_mask is None:
        x_mask = torch.ones_like(x_C[:, :, 0]).bool()
    if z_mask is None:
        z_mask = torch.ones_like(z_C[:, :, 0]).bool()
    dists = norm(x_C[:, None, :, :] - z_C[:, :, None, :], -1)  # B,M,N
    dist_recip = 1.0 / (dists + 1e-8)
    dist_recip = dist_recip * x_mask[:, None, :]
    normalizer = dist_recip.sum(-1)
    weights = dist_recip / normalizer[..., None]
    z_F = torch.bmm(weights, x_F)
    return z_F


def interpolate_knn(x_C, x_F, z_C, K, x_mask=None, z_mask=None):
    """

    :param x_C: N,3
    :param x_F:N,C
    :param z_C:M,3
    :param K:int
    :param x_mask:
    :param z_mask:
    :return:
    """
    import torch_cluster
    N = x_C.shape[0]
    C = x_F.shape[-1]
    M = z_C.shape[1]
    L = z_C.shape[0]
    if x_mask is None:
        x_mask = torch.ones_like(x_C[..., 0]).bool()
    if z_mask is None:
        z_mask = torch.ones_like(z_C[..., 0]).bool()
    z_C_ = z_C.reshape(-1, 3)
    idxs = torch_cluster.knn(x_C, z_C_, K)
    assert idxs.shape[1] == z_C_.shape[0] * K
    idxs = idxs[1].reshape(z_C_.shape[0], K)
    idxs = idxs.reshape(*z_C.shape[:-1], K)
    # x_F[idxs.reshape(-1)].reshape(-1,M,K,C)
    dists = norm(x_C[idxs.reshape(-1)].reshape(L, M, K, 3), -1)
    dist_recip = 1.0 / (dists + 1e-8)
    # dist_recip = dist_recip * x_mask[:, None, :]
    normalizer = dist_recip.sum(-1)
    weights = dist_recip / normalizer[..., None]
    idx_F = x_F[idxs.reshape(-1)].reshape(L, M, K, C)
    dst_F = (idx_F * weights[..., None]).sum(-2)
    return dst_F


def interpolate_scene_flow_from_voxel_matching(pred):
    voxel_size = pred['voxel_size']
    input0 = pred['input0']
    coord0 = input0.C[:, :3]  # R

    deformed_points_pred = pred['deformed_points_pred'][0] / voxel_size
    stage2_pred_voxel_offsets = deformed_points_pred - coord0[:, :3]
    xC = torch.cat((coord0, torch.zeros_like(coord0[:, 0:1])), dim=1)
    x = ts.SparseTensor(stage2_pred_voxel_offsets, xC)
    c0 = pred['coord0'][0] / voxel_size
    c0 = torch.cat((c0, torch.zeros_like(c0[:, 0:1])), dim=1)
    z = ts.PointTensor(c0, c0)
    znew = voxel_to_point(x, z)
    pred_scene_flow = znew.F
    pred_scene_flow = pred_scene_flow * voxel_size
    return pred_scene_flow


def pad_batch_idx(st, batch_idx=0):
    if isinstance(st, (ts.SparseTensor, ts.PointTensor)):
        if st.C.shape[1] == 4:
            return st
        bi = torch.full_like(st.C[:, 0:1], batch_idx)
        st.C = torch.cat((st.C, bi), dim=1)
    elif isinstance(st, torch.Tensor):
        assert st.shape[1] == 3
        st = torch.cat([st, torch.full_like(st[:, 0:1], batch_idx)], dim=1)
    else:
        raise NotImplementedError()
    return st


def remove_batch_idx(x):
    assert isinstance(x, (ts.SparseTensor, ts.PointTensor))
    assert x.C.shape[1] == 4
    x.C = x.C[:, :3]
    return x


def ptidx2voxelidx_to_voxelidx2ptidxs_py(M0, K, pointwise_pixelloc_to_voxelidx):
    evaltime = EvalTime(disable=True)
    evaltime('begin')
    voxelidx_to_ptidxs = -torch.ones(M0, K).long().to(device=pointwise_pixelloc_to_voxelidx.device)
    cnts = torch.zeros(M0).long()
    for i, ptv in enumerate(pointwise_pixelloc_to_voxelidx):
        if cnts[ptv] < 20:
            voxelidx_to_ptidxs[ptv, cnts[ptv]] = i
            cnts[ptv] += 1
    evaltime('ptidx2voxelidx_to_voxelidx2ptidxs_py')
    # voxelidx_to_ptidxs[pointwise_pixelloc_to_voxelidx] = torch.arange(M).to(device=pointwise_pixelloc_to_voxelidx.device)
    return voxelidx_to_ptidxs


def ptidx2voxelidx_to_voxelidx2ptidxs_c(M0, K, pointwise_pixelloc_to_voxelidx, random_sample=False):
    evaltime = EvalTime(disable=True)
    evaltime('begin')
    from NeuralNRT._C import ptidx2voxelidx_to_voxelidx2ptidxs_v2
    # voxelidx_to_ptidxs = np.zeros((0), dtype=np.int32)
    pptv = to_array(pointwise_pixelloc_to_voxelidx)
    # ptidx2voxelidx_to_voxelidx2ptidxs(pptv, voxelidx_to_ptidxs, M0, K)
    voxelidx_to_ptidxs2 = np.zeros((0), dtype=np.int32)
    if random_sample:
        perm = np.random.permutation(pptv.shape[0])
    else:
        perm = np.arange(pptv.shape[0], dtype=int)
    pptv_perm = pptv[perm]
    ptidx2voxelidx_to_voxelidx2ptidxs_v2(pptv_perm, voxelidx_to_ptidxs2, perm, M0, K)
    voxelidx_to_ptidxs = torch.from_numpy(voxelidx_to_ptidxs2).to(device=pointwise_pixelloc_to_voxelidx.device).long()
    evaltime('ptidx2voxelidx_to_voxelidx2ptidxs_c')
    return voxelidx_to_ptidxs


def sub_sparse_tensor(x: ts.SparseTensor, batch_idx):
    keep = x.C[:, -1] == batch_idx
    return ts.SparseTensor(x.F[keep], x.C[keep])


def ptidx2voxelidx_to_voxelidx2ptidxs_v2_py(pptv_perm, voxelidx_to_ptxidxs2, perm, M0, K):
    voxelidx_to_ptidxs = -np.ones((M0, K))
    cnts = np.zeros((M0), dtype=int)
    for i in range(pptv_perm.shape[0]):
        ptv = int(pptv_perm[i])
        reali = perm[i]
        if cnts[ptv] < K:
            voxelidx_to_ptidxs[ptv, cnts[ptv]] = reali
            cnts[ptv] += 1
    return voxelidx_to_ptidxs


def clone_point_tensor(z):
    assert isinstance(z, ts.PointTensor)
    new_z = ts.PointTensor(clone(z.F), clone(z.C))
    new_z.additional_features = clone(z.additional_features)
    new_z.idx_query = clone(z.idx_query)
    new_z.weights = clone(z.weights)
    return new_z


# import numpy as np
# import torch
#
# __all__ = ['KernelRegion']


class KernelRegion:
    def __init__(self,
                 kernel_size: int = 3,
                 tensor_stride: int = 1,
                 dilation: int = 1,
                 dim=[0, 1, 2]) -> None:
        self.kernel_size = kernel_size
        self.tensor_stride = tensor_stride
        self.dilation = dilation

        if not isinstance(kernel_size, (list, tuple)):
            if kernel_size % 2 == 0:
                # even
                region_type = 0
            else:
                # odd
                region_type = 1

            self.region_type = region_type

            single_offset = (
                    np.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1) *
                    tensor_stride * dilation).tolist()

            x_offset = single_offset if 0 in dim else [0]
            y_offset = single_offset if 1 in dim else [0]
            z_offset = single_offset if 2 in dim else [0]

            if self.region_type == 1:
                kernel_offset = [[x, y, z] for z in z_offset for y in y_offset
                                 for x in x_offset]
            else:
                kernel_offset = [[x, y, z] for x in x_offset for y in y_offset
                                 for z in z_offset]
            kernel_offset = np.array(kernel_offset)
            self.kernel_offset = torch.from_numpy(kernel_offset).int()
        else:
            if dim == [0, 1, 2] and len(kernel_size) == 3:
                kernel_x_size = kernel_size[0]
                kernel_y_size = kernel_size[1]
                kernel_z_size = kernel_size[2]

                x_offset = (np.arange(-kernel_x_size // 2 + 1,
                                      kernel_x_size // 2 + 1) * tensor_stride *
                            dilation).tolist()
                y_offset = (np.arange(-kernel_y_size // 2 + 1,
                                      kernel_y_size // 2 + 1) * tensor_stride *
                            dilation).tolist()
                z_offset = (np.arange(-kernel_z_size // 2 + 1,
                                      kernel_z_size // 2 + 1) * tensor_stride *
                            dilation).tolist()

                kernel_offset = [[x, y, z] for x in x_offset for y in y_offset
                                 for z in z_offset]

                kernel_offset = np.array(kernel_offset)
                self.kernel_offset = torch.from_numpy(kernel_offset).int()
            else:
                raise NotImplementedError

    def get_kernel_offset(self):
        return self.kernel_offset


def sparse_to_dense_torch(locs, values, dim=None, default_val=0.0, device='cuda'):
    if dim is None:
        mxyz = locs.min(0).values
        Mxyz = locs.max(0).values
        dim = (Mxyz - mxyz + 1).tolist()
    dense = torch.full(dim, float(default_val), device=device)
    assert torch.all(locs >= 0)
    if locs.shape[0] > 0:
        assert locs[:, 0].max() < dim[0]
        assert locs[:, 1].max() < dim[1]
        assert locs[:, 2].max() < dim[2]
        dense[locs[:, 0], locs[:, 1], locs[:, 2]] = values
    return dense


def sparse_to_dense_channel(locs, values, dim, c, default_val, device):
    dense = torch.full([dim[0], dim[1], dim[2], c], float(default_val), device=device)
    assert torch.all(locs >= 0)
    if locs.shape[0] > 0:
        assert locs[:, 0].max() < dim[0]
        assert locs[:, 1].max() < dim[1]
        assert locs[:, 2].max() < dim[2]
        dense[locs[:, 0], locs[:, 1], locs[:, 2]] = values
    return dense


def sparse_tensor_intersection(st0C, st1C, st0F=None, st1F=None):
    st0C = st0C.long()
    st1C = st1C.long()
    xyzm0 = st0C.min(0).values
    xyzM0 = st0C.max(0).values
    xyzm1 = st1C.min(0).values
    xyzM1 = st1C.max(0).values
    xyzm = torch.min(xyzm0, xyzm1)
    xyzM = torch.max(xyzM0, xyzM1)
    dim_list = (xyzM - xyzm + 1).tolist()[:3]
    if st0F is None:
        st0F = torch.ones_like(st0C[:, 0:1]).float()
    if st1F is None:
        st1F = torch.ones_like(st1C[:, 0:1]).float()

    default_values = 0.0
    st0C_positive = st0C - xyzm
    st1C_positive = st1C - xyzm
    st0_volume = sparse_to_dense_channel(st0C_positive.long()[:, :3], st0F, dim_list,
                                         st0F.shape[1], default_values, st0F.device)
    st1_volume = sparse_to_dense_channel(st1C_positive.long()[:, :3], st1F, dim_list,
                                         st1F.shape[1], default_values, st1F.device)
    intersection_coords_positive = torch.nonzero(
        (st0_volume != 0).any(-1) & (st1_volume != 0).any(-1), as_tuple=False)
    intersection_coords = intersection_coords_positive + xyzm
    return intersection_coords


# def sparse_tensor_union(st0C, st1C, st0F=None, st1F=None):
#     xyzm0 = st0C.min(0).values
#     xyzM0 = st0C.max(0).values
#     xyzm1 = st1C.min(0).values
#     xyzM1 = st1C.max(0).values
#     xyzm = torch.min(xyzm0, xyzm1)
#     xyzM = torch.max(xyzM0, xyzM1)
#     dim_list = (xyzM - xyzm + 1).tolist()[:3]
#     if st0F is None:
#         st0F = torch.ones_like(st0C[:, 0:1])
#     if st1F is None:
#         st1F = torch.ones_like(st1C[:, 0:1])
#
#     default_values = 0
#     st0C_positive = st0C - xyzm
#     st1C_positive = st1C - xyzm
#     st0_volume = sparse_to_dense_channel(st0C_positive.long()[:, :3], st0F, dim_list,
#                                          st0F.shape[1], default_values, st0F.device)
#     st1_volume = sparse_to_dense_channel(st1C_positive.long()[:, :3], st1F, dim_list,
#                                          st1F.shape[1], default_values, st1F.device)
#     updated_coords_positive = torch.nonzero(
#         (st0_volume != 0).any(-1) | (st1_volume != 0).any(-1), as_tuple=False)

def merge_sparse_tensors(st0C, st1C):
    if st0C.shape[0] == 0:
        return st1C[:, :3], st0C[:, :3], st1C[:, :3]
    if st0C.shape[1] == 3:
        st0C = pad_batch_idx(st0C, 0)
    if st1C.shape[1] == 3:
        st1C = pad_batch_idx(st1C, 0)
    st0_hash = spnn.sphash(st0C)
    st1_hash = spnn.sphash(st1C)
    int1d, ind1, ind2 = intersect1d_pytorch(st0_hash, st1_hash, return_indices=True)
    keep = torch.ones(st0C.shape[0]).bool()
    keep[ind1] = False
    st0C = st0C[keep]
    merged_stC = torch.cat([st0C, st1C])
    merged_stC = merged_stC[:, :3]
    return merged_stC, st0C[:, :3], st1C[:, :3]


# timer = Timer()


def dilate_sparse_tensor(stC, dilate_r, dilate_num=26):
    # evaltime=EvalTime()
    # evaltime('')
    # timer.tic()
    if dilate_num == 2:
        warnings.warn("dilate num should be 26.")
    if dilate_r == 0:
        return torch.empty([0, 3], dtype=torch.int, device='cuda')
    stC3 = stC[:, :3]
    if dilate_num == 26:
        el = [0]
        for i in range(dilate_r):
            el.append(-i - 1)
            el.append(i + 1)
        # print(el)
        prod = list(product(el, el, el))
        prod = torch.tensor(prod)[1:]

        dilate_stC = stC3.unsqueeze(1).repeat(1, prod.shape[0], 1)
        dilate_stC = dilate_stC + prod[None].to(dilate_stC.device).int()
        dilate_stC = dilate_stC.reshape(-1, 3)
    else:
        dilate_stC = torch.cat([stC3 - 1, stC3 + 1])
    dilate_stC4 = pad_batch_idx(dilate_stC)
    hash_d = spnn.sphash(dilate_stC4)
    ar1, inv_ind1 = torch.unique(hash_d, return_inverse=True)
    perm = torch.arange(inv_ind1.size(0), dtype=inv_ind1.dtype, device=inv_ind1.device)
    inverse, perm = inv_ind1.flip([0]), perm.flip([0])
    ind1 = inverse.new_empty(ar1.size(0)).scatter_(0, inverse, perm)
    dilate_stC = dilate_stC[ind1]
    # timer.toc()
    # print('dilate', timer.average_time)
    return dilate_stC


def marching_cubes_for_sparse_tensor(stC, tsdf, voxel_size, occ=None, return_trimesh=False, backend='skimage',
                                     device='cuda'):
    if occ is None:
        keep = (tsdf > -1) & (tsdf < 1)
    else:
        keep = occ > 0
    tsdf = tsdf.reshape(-1)[keep]
    stC = stC[keep]
    coord_int = stC[:, :3]
    xyzm = coord_int.min(0).values
    xyzM = coord_int.max(0).values
    coord_int_positive = (coord_int - coord_int.min(0).values).long()
    dense_sdf = sparse_to_dense_torch(coord_int_positive, tsdf, (xyzM - xyzm + 2).tolist(), 1.0,
                                      tsdf.device)  # todo: default?
    if backend == 'skimage':
        verts, faces, normals, values = marching_cubes(to_array(dense_sdf), level=0, spacing=[voxel_size] * 3)
        verts = verts + xyzm.cpu().numpy() * voxel_size
        verts = torch.from_numpy(verts).float().to(device=device)
        faces = torch.from_numpy(faces.copy()).long().to(device=device)
        # normals = torch.from_numpy(normals.copy()).float().cuda()
        # values = torch.from_numpy(values.copy()).float().cuda()
    elif backend == 'torchmcubes':
        from torchmcubes import marching_cubes as marching_cubes_cuda
        verts, faces = marching_cubes_cuda(dense_sdf, 0.0)
    else:
        raise NotImplementedError()
    if not return_trimesh:
        return verts, faces
    else:
        return trimesh.Trimesh(to_array(verts), to_array(faces))


def fill_holes_in_sparse_tensor(st, device='cuda'):
    C = st.C[:, :3]
    feat = st.F
    min = C.min(0).values[None]
    C = C - min
    mxyz = C.min(0).values
    Mxyz = C.max(0).values
    dim = (Mxyz - mxyz + 1).tolist()
    C_dense = sparse_to_dense_channel(C.long(), feat, dim, feat.shape[1], 0, device=device).permute(3, 0, 1, 2)
    # C_dense = sparse_to_dense_torch(C.long(), torch.ones_like(C[:, 0].float()))
    c, d, h, w = C_dense.size()
    x_ = torch.arange(w).view(1, 1, -1).expand(d, h, -1)
    y_ = torch.arange(h).view(1, -1, 1).expand(d, -1, w)  # h,w
    z_ = torch.arange(d).view(-1, 1, 1).expand(-1, h, w)  # h,w
    grid = torch.stack([x_, y_, z_], dim=0).float().to(device=device)  # 3,d,h,w
    grid[0, :, :, :] = 2 * grid[0, :, :, :] / (w - 1) - 1
    grid[1, :, :, :] = 2 * grid[1, :, :, :] / (h - 1) - 1
    grid[2, :, :, :] = 2 * grid[2, :, :, :] / (d - 1) - 1
    grid = grid[None].permute(0, 2, 3, 4, 1)
    gs_res = F.grid_sample(C_dense[None], grid, mode='bilinear', align_corners=False)
    new_C = (gs_res[0] != 0).any(dim=0).nonzero(as_tuple=False)
    new_F = gs_res[0, :, new_C[:, 0], new_C[:, 1], new_C[:, 2]].T
    new_C = new_C + min
    new_C = pad_batch_idx(new_C)
    new_st = ts.SparseTensor(new_F, new_C)
    return new_st


def empty_sparse_tensor(feature_shape=[0, 1], C_dim=4, device='cuda'):
    return ts.SparseTensor(torch.empty(feature_shape).float().to(device),
                           torch.empty([0, C_dim], dtype=torch.int, device=device))


def cat_sparse_tensors(st0, st1):
    return ts.SparseTensor(torch.cat([st0.F, st1.F]), torch.cat([st0.C, st1.C]))
