import pyrender
import tqdm
import numpy as np
import os

import matplotlib.pyplot as plt
import torch
import trimesh
from dl_ext.timer import EvalTime
from dl_ext.vision_ext.datasets.kitti.structures import Calibration
from pyrender import Node
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader, Textures, TexturesVertex, PerspectiveCameras, SfMPerspectiveCameras, get_world_to_view_transform,
)
from pytorch3d.renderer.mesh.renderer import MeshRendererWithFragments
from pytorch3d.structures import Meshes
from tqdm import trange

from disprcnn.utils.pn_utils import min_max, ptp, to_array

# os.system('mkdir -p data/cow_mesh')
# os.system('wget -P data/cow_mesh https://dl.fbaipublicfiles.com/pytorch3d/data/cow_mesh/cow.obj')
# os.system('wget -P data/cow_mesh https://dl.fbaipublicfiles.com/pytorch3d/data/cow_mesh/cow.mtl')
# os.system('wget -P data/cow_mesh https://dl.fbaipublicfiles.com/pytorch3d/data/cow_mesh/cow_texture.png')

# Setup


# # Set paths
# DATA_DIR = "./data"
# obj_filename = os.path.join(DATA_DIR, "cow_mesh/cow.obj")
#
# # Load obj file
# mesh = load_objs_as_meshes([obj_filename], device=device)
#

# ## 2. Create a renderer
#
# A renderer in PyTorch3D is composed of a **rasterizer** and a **shader** which each have a number of subcomponents such as a **camera** (orthographic/perspective). Here we initialize some of these components and use default values for the rest.
#
# In this example we will first create a **renderer** which uses a **perspective camera**, a **point light** and applies **Phong shading**. Then we learn how to vary different components using the modular API.
from disprcnn.utils.utils_3d import canonical_to_camera_np, world_coordinate_to_camera_coordinate, matrix_3x4_to_4x4, rotx, \
    roty_np, rotx_np, create_center_radius, canonical_to_camera, transform_points


def render_mesh_api(verts, faces, focal_length, dist=1.57, batch_size=20, image_size=512, textures=None):
    if isinstance(image_size, int):
        image_size = ((image_size, image_size),)
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    if textures is None:
        verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
        textures = TexturesVertex(verts_rgb)

    mesh = Meshes(verts[None], faces[None], textures=textures).to(device=device)
    max_extent = ptp(mesh.get_bounding_boxes()[0], dim=1).max()

    # Get a batch of viewing angles.
    elev = torch.linspace(-89, 89, batch_size)
    azim = torch.linspace(-179, 179, batch_size)
    elev_interval = 180.0 / batch_size
    elev_offset = elev_interval * 0.4 * torch.rand_like(elev)
    elev = elev_offset + elev
    elev[0] = -89
    elev[-1] = 89
    azim_interval = 360.0 / batch_size
    azim_offset = azim_interval * 0.4 * torch.rand_like(azim)
    azim = azim_offset + azim
    azim[0] = -179
    azim[-1] = 179
    # All the cameras helper methods support mixed type inputs and broadcasting. So we can
    # view the camera from the same distance and specify dist=2.7 as a float,
    # and then specify elevation and azimuth angles for each viewpoint as tensors.
    # dist = 1.75
    R, T = look_at_view_transform(dist=dist * max_extent, elev=elev, azim=azim)
    wtvt = get_world_to_view_transform(R=R, T=T)
    # focal_length = 600.0
    cameras = PerspectiveCameras(focal_length=focal_length,
                                 principal_point=((image_size[0][0] / 2, image_size[0][1] / 2),),
                                 # K=K,
                                 R=R, T=T,
                                 device=device,
                                 in_ndc=False,
                                 image_size=image_size
                                 )

    raster_settings = RasterizationSettings(
        image_size=image_size[0],
        blur_radius=0.0,
        faces_per_pixel=1,
    )

    lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

    renderer = MeshRendererWithFragments(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(
            device=device,
            cameras=cameras,
            lights=lights
        )
    )
    # todo: keep rasterizer only?

    meshes = mesh.extend(batch_size)
    # meshes = meshes.update_padded(wtvt.transform_points(verts).cuda())
    lights.location = torch.tensor([[0.0, 0.0, -3.0]], device=device)
    outputs = renderer(meshes, cameras=cameras, lights=lights)
    depths = outputs[1].zbuf[:, :, :, 0]
    images = outputs[0][..., :3]
    return images.cpu(), depths.cpu(), cameras, meshes


def pyrender_api(verts, faces, focal_length, dist=1.57, batch_size=20, image_size=512, textures=None,
                 round2=False, max_elev=89, min_elev=0, noise=True):
    import pyrender
    max_extent = max(verts.max(0).values - verts.min(0).values)
    elev = torch.linspace(-max_elev, max_elev, batch_size)
    elev_interval = 2 * max_elev / batch_size
    if noise:
        elev_offset = elev_interval * 0.4 * torch.rand_like(elev)
    else:
        elev_offset = 0
    elev = elev_offset + elev
    elev[0] = -max_elev
    elev[-1] = max_elev
    elev[(elev < min_elev) & (elev > 0)] = min_elev
    elev[(-min_elev < elev) & (elev < 0)] = -min_elev

    if not round2:
        azim = torch.linspace(-179, 179, batch_size)
        azim_interval = 360.0 / batch_size
        azim_offset = azim_interval * 0.4 * torch.rand_like(azim)
        azim = azim_offset + azim
        azim[0] = -179
        azim[-1] = 179
    else:
        azim1 = torch.linspace(-179, 179, batch_size // 2)
        azim2 = torch.linspace(181, 539, batch_size // 2)
        azim = torch.cat([azim1, azim2])
        azim_interval = 360.0 / batch_size
        azim_offset = azim_interval * 0.4 * torch.rand_like(azim)
        if not noise:
            azim_offset = 0
        azim = azim_offset + azim
        azim[0] = -179
        azim[-1] = 539

    gaussian_noise = torch.randn(batch_size) * 0.1
    dist = dist * max_extent + gaussian_noise
    R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim)
    if isinstance(image_size, int):
        image_size = (image_size, image_size)
    cy, cx = image_size[0] / 2, image_size[1] / 2
    camera = pyrender.IntrinsicsCamera(fx=focal_length, fy=focal_length, cx=cx, cy=cy)
    coord_convert = np.eye(4)
    coord_convert[:3, :3] = roty_np(np.pi / 2) @ rotx_np(- np.pi / 2)
    coord_convert[:3, :3] = rotx_np(- np.pi)
    light = pyrender.SpotLight(color=np.ones(3), intensity=3.0,
                               innerConeAngle=np.pi / 16.0)
    scene = pyrender.Scene()
    scene.add(camera, pose=coord_convert)
    scene.add(light, pose=coord_convert)
    RT = torch.cat((R, T[..., None]), -1)
    poses = matrix_3x4_to_4x4(RT)
    r = pyrender.OffscreenRenderer(image_size[1], image_size[0])
    flags = pyrender.constants.RenderFlags.DEPTH_ONLY
    depths = []
    for pose in tqdm.tqdm(poses, leave=False):
        m = trimesh.Trimesh(canonical_to_camera_np(verts, pose=pose), faces)
        mesh = pyrender.Mesh.from_trimesh(m)
        node = Node(mesh=mesh)
        scene.add_node(node)
        depth = r.render(scene, flags)
        scene.remove_node(node)
        depths.append(depth)
    depths = torch.from_numpy(np.stack(depths))
    return depths, poses


def pyrender_ring_api(verts, faces, focal_length, dist=1.57, image_size=512, nelev=18, nazim=12, max_elev=80):
    import pyrender
    max_extent = max(verts.max(0).values - verts.min(0).values)
    # nelev, nazim = 18, 12
    # max_elev = 80
    # dist = 1.57

    elev = np.linspace(-max_elev, max_elev, nelev)
    all_RT = []
    for e in elev:
        RT = torch.from_numpy(create_center_radius(dist=dist, nrad=nazim, angle_z=e))
        all_RT.append(RT)
    all_RT = torch.cat(all_RT, 0)
    poses = matrix_3x4_to_4x4(all_RT)

    if isinstance(image_size, int):
        image_size = (image_size, image_size)
    cy, cx = image_size[0] / 2, image_size[1] / 2
    camera = pyrender.IntrinsicsCamera(fx=focal_length, fy=focal_length, cx=cx, cy=cy)
    coord_convert = np.eye(4)
    coord_convert[:3, :3] = rotx_np(-np.pi)
    light = pyrender.SpotLight(color=np.ones(3), intensity=3.0,
                               innerConeAngle=np.pi / 16.0)
    scene = pyrender.Scene()
    scene.add(camera, pose=coord_convert)
    scene.add(light, pose=coord_convert)
    r = pyrender.OffscreenRenderer(image_size[1], image_size[0])
    flags = pyrender.constants.RenderFlags.DEPTH_ONLY
    depths = []
    for pose in tqdm.tqdm(poses, leave=False, desc='rendering depth'):
        m = trimesh.Trimesh(canonical_to_camera_np(verts, pose=pose), faces)
        mesh = pyrender.Mesh.from_trimesh(m)
        node = Node(mesh=mesh)
        scene.add_node(node)
        depth = r.render(scene, flags)
        scene.remove_node(node)
        depths.append(depth)
    depths = torch.from_numpy(np.stack(depths))
    return depths, poses


@torch.no_grad()
def pytorch3d_render_ring_api(verts, faces, focal_length, cx=None, cy=None, dist=1.57, image_size=512, nelev=18,
                              nazim=12, max_elev=80, bin_size=None):
    # nelev, nazim = 18, 12
    # max_elev = 80

    elev = np.linspace(-max_elev, max_elev, nelev)
    all_RT = []
    for e in elev:
        RT = torch.from_numpy(create_center_radius(dist=dist, nrad=nazim, angle_z=e))
        all_RT.append(RT)
    all_RT = torch.cat(all_RT, 0)
    poses = matrix_3x4_to_4x4(all_RT)

    # image_width, image_height = 1600, 1200
    if not isinstance(focal_length, (tuple, list)):
        focal_length = (focal_length, focal_length)
    if cx is None:
        cx = image_size[1] / 2
    if cy is None:
        cy = image_size[0] / 2
    # cy, cx = image_size[0] / 2, image_size[1] / 2
    principal_point = (cx, cy)
    device = 'cuda'
    # coord_convert = np.eye(4)
    # coord_convert[:3, :3] = rotx_np(-np.pi)
    cameras = PerspectiveCameras(focal_length=(focal_length,),
                                 principal_point=(principal_point,),
                                 device=device,
                                 image_size=((image_size[1], image_size[0]),),
                                 # R=torch.from_numpy(rotx_np(-np.pi)).float().cuda()
                                 )
    raster_settings = RasterizationSettings(
        image_size=(image_size[0], image_size[1]),
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=bin_size,
        max_faces_per_bin=None
    )
    rasterizer = MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings
    )
    # transform verts
    verts_hom = Calibration.cart_to_hom(verts)
    t_verts_hom = torch.bmm(verts_hom[None].repeat(poses.shape[0], 1, 1), poses.permute(0, 2, 1).float())
    t_verts = t_verts_hom[:, :, :3] / t_verts_hom[:, :, 3:]
    # t_verts[:, :2] *= -1
    t_verts[:, :, :2] *= -1
    # faces = faces[None].repeat(poses.shape[0], 1, 1).long().cuda()
    # make small batches
    sbsz = 20
    depths = []
    evaltime = EvalTime(disable=True)
    evaltime('')
    for i in trange(0, t_verts.shape[0], sbsz):
        mt_verts = t_verts[i: i + sbsz]
        mt_verts = mt_verts.cuda()
        mt_faces = faces[None].repeat(mt_verts.shape[0], 1, 1).long().cuda()
        mesh = Meshes(mt_verts, mt_faces)
        evaltime('prepare mesh')
        fragments = rasterizer(mesh)
        evaltime('ras')
        depth = fragments.zbuf[..., 0].detach()
        depths.append(depth)
        evaltime('append')
    depths = torch.cat(depths).cpu()
    return depths, poses


class PyrenderRenderMeshApiHelper:
    _renderer = None

    # def __init__(self):
    #     self._renderer = None

    @staticmethod
    def get_renderer(H, W):
        if PyrenderRenderMeshApiHelper._renderer is None:
            PyrenderRenderMeshApiHelper._renderer = pyrender.OffscreenRenderer(W, H)
        return PyrenderRenderMeshApiHelper._renderer


def pyrender_render_mesh_api(mesh: trimesh.Trimesh, object_pose, H, W, K):
    evaltime = EvalTime(disable=True)
    evaltime('')
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    fu, fv, cu, cv = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    evaltime('1')
    camera = pyrender.IntrinsicsCamera(fx=fu, fy=fv, cx=cu, cy=cv)
    coord_convert = np.eye(4)
    coord_convert[:3, :3] = rotx(-np.pi)
    light = pyrender.SpotLight(color=np.ones(3), intensity=3.0,
                               innerConeAngle=np.pi / 16.0)
    object_pose = to_array(object_pose)
    vertices = transform_points(mesh.vertices, object_pose)
    mesh = trimesh.Trimesh(vertices, mesh.faces)
    evaltime('3')
    mesh = pyrender.Mesh.from_trimesh(mesh)
    evaltime('4')
    scene = pyrender.Scene()
    scene.add(mesh)
    scene.add(camera, pose=coord_convert)
    scene.add(light, pose=coord_convert)
    flags = pyrender.constants.RenderFlags.DEPTH_ONLY
    evaltime('5')
    # r = pyrender.OffscreenRenderer(W, H)
    r = PyrenderRenderMeshApiHelper.get_renderer(H, W)
    evaltime('6')
    depth = r.render(scene, flags)
    evaltime('7')
    mask = depth > 0
    evaltime('8')
    return mask
