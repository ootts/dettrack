import os
import os.path as osp
from typing import List

import trimesh
from trimesh import Trimesh

default_root = osp.expanduser('~/Datasets/Apollo/3d_car_instance_sample/3d_car_instance_sample/car_models_ply/')


class ApolloCarInstanceSampleDB:
    def __init__(self, root=default_root):
        self.root = root

    def load_meshlist(self, absolute_path=False, remove_suffix=False):
        """
        load all mesh names
        :param absolute_path: return absolute path for each file if True
        :param remove_suffix: remove file extension if True, only work when absolute_path is False
        :return:
        """
        results = []
        for file in os.listdir(self.root):
            if absolute_path:
                results.append(osp.join(self.root, file))
            elif remove_suffix:
                results.append(file.split('.')[0])
            else:
                results.append(file)
        return results

    def load_mesh(self, name: str, move_to_center: bool = True) -> Trimesh:
        if not name.endswith('ply'):
            name = name + '.ply'
        try:
            mesh: Trimesh = trimesh.load(name)
        except:
            mesh: Trimesh = trimesh.load(osp.join(self.root, name))
        if move_to_center:
            mesh.vertices -= mesh.bounding_box.centroid
        # mesh.vertices[:, 2] *= -1
        return mesh

    def load_all_meshes(self, move_to_center: bool = True) -> List[Trimesh]:
        meshlist = self.load_meshlist(absolute_path=True)
        all_meshes = []
        for ml in meshlist:
            mesh = self.load_mesh(ml, move_to_center)
            all_meshes.append(mesh)
        return all_meshes
