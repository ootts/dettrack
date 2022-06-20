import numpy as np
import open3d as o3d

from disprcnn.utils.utils_3d import transform_points
from disprcnn.utils.vis3d_ext import Vis3D


def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result


def fpfh_registration_api(pts0, pts1, voxel_size, icp=False, icp_thresh=None, dbg=False):
    vis3d = Vis3D(
        xyz_pattern=('x', 'y', 'z'),
        out_folder="dbg",
        sequence="fpfh_registration_api",
        auto_increase=True,
        enable=dbg,
    )
    vis3d.add_point_cloud(pts0, name='pts0')
    vis3d.add_point_cloud(pts1, name='pts1')

    pcd0 = o3d.geometry.PointCloud()
    pcd0.points = o3d.utility.Vector3dVector(pts0)
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(pts1)
    source_down, source_fpfh = preprocess_point_cloud(pcd0, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(pcd1, voxel_size)
    result_ransac = execute_global_registration(source_down, target_down,
                                                source_fpfh, target_fpfh,
                                                voxel_size)
    vis3d.add_point_cloud(transform_points(pts0, result_ransac.transformation), name='tpts0')
    # transformation =
    if icp:
        result_ransac = o3d.pipelines.registration.registration_icp(
            source_down, target_down, icp_thresh, result_ransac.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane())
        # transformation = result.transformation
    return result_ransac
