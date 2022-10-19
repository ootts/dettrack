import math

import imageio
import numpy as np
import torch
from dl_ext.vision_ext.transforms import imagenet_normalize
from disprcnn.modeling.models.psmnet_vanilla.stackhourglass import PSMNet
from torchvision.transforms import transforms
from disprcnn.utils.vis3d_ext import Vis3D

from disprcnn.utils import utils_3d


class PSMNetHelper:
    model = None

    @staticmethod
    def getinstance(pretrained_model=None):
        if PSMNetHelper.model is None:
            state_dict = torch.load(pretrained_model)['state_dict']
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            model = PSMNet(192)
            model.load_state_dict(state_dict)
            model.cuda()
            model.eval()
            PSMNetHelper.model = model
        return PSMNetHelper.model


@torch.no_grad()
def psmnet_vanilla_api(left_img_path, right_img_path, pretrained_model=None):
    left = imageio.imread(left_img_path)
    right = imageio.imread(right_img_path)
    if len(left.shape) == 2:
        left = np.repeat(left[:, :, None], 3, axis=-1)
    if len(right.shape) == 2:
        right = np.repeat(right[:, :, None], 3, axis=-1)
    transform = transforms.Compose([
        transforms.ToTensor(),
        imagenet_normalize
    ])
    oh, ow = left.shape[:2]
    h = math.ceil(oh / 16) * 16
    w = math.ceil(ow / 16) * 16
    img2 = transform(left)
    img3 = transform(right)
    left_input = torch.zeros((1, 3, h, w)).cuda()
    right_input = torch.zeros((1, 3, h, w)).cuda()
    left_input[:, :, :oh, :ow] = img2
    right_input[:, :, :oh, :ow] = img3
    model = PSMNetHelper.getinstance(pretrained_model)
    output = model(left_input, right_input)
    output = output.cpu()[0, :oh, :ow].numpy()
    return output
