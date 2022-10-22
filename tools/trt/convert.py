import os
# import torchvision.models as models
#
# resnext50_32x4d = models.resnext50_32x4d(pretrained=True)
#
# import torch
#
# BATCH_SIZE = 64
# dummy_input = torch.randn(BATCH_SIZE, 3, 224, 224)
#
# import torch.onnx
#
# torch.onnx.export(resnext50_32x4d, dummy_input, "tmp/resnet50_onnx_model.onnx", verbose=False)
from disprcnn.config import cfg
from PIL import Image
from io import BytesIO
import requests

from disprcnn.engine.defaults import default_argument_parser
from disprcnn.structures.bounding_box import BoxList
from disprcnn.trainer.build import build_trainer
from disprcnn.utils.comm import get_rank
from disprcnn.utils.logger import setup_logger

# output_image = "tmp/input.ppm"
#
# # Read sample image input and save it in ppm format
# print("Exporting ppm image {}".format(output_image))
# response = requests.get("https://pytorch.org/assets/images/deeplab1.png")
# with Image.open(BytesIO(response.content)) as img:
#     ppm = Image.new("RGB", img.size, (255, 255, 255))
#     ppm.paste(img, mask=img.split()[3])
#     ppm.save(output_image)

import torch
import torch.nn as nn

output_onnx = "tmp/disprcnn.onnx"


# FC-ResNet101 pretrained model from torch-hub extended with argmax layer
class FCN_ResNet101(nn.Module):
    def __init__(self, model, calib):
        super(FCN_ResNet101, self).__init__()
        self.model = model
        self.calib = calib
        # self.transforms = transforms
        self.global_step = 0

    def inputs_to_dps(self, inputs):
        device = inputs.device
        left_original_img, right_original_img, left_img, right_img = torch.split(inputs, 1, dim=0)
        left_original_img = left_original_img.permute(0, 2, 3, 1)
        right_original_img = right_original_img.permute(0, 2, 3, 1)
        left_img = left_img[:, :, :300:600]
        right_img = right_img[:, :, :300:600]
        left_target = BoxList(torch.empty([0, 4], device=device, dtype=torch.float), (1, 1))
        right_target = BoxList(torch.empty([0, 4], device=device, dtype=torch.float), (1, 1))
        left_target.add_field("calib", self.calib)

        dps = {
            'original_images': {
                'left': left_original_img,
                'right': right_original_img,
            },
            'images': {
                'left': left_img,
                'right': right_img,
            },
            'targets': {
                'left': [left_target],
                'right': [right_target]
            },
            'height': torch.tensor([left_original_img.shape[1]], device=device, dtype=torch.long),
            'width': torch.tensor([left_original_img.shape[2]], device=device, dtype=torch.long),
            'index': torch.tensor([self.global_step], device=device, dtype=torch.long),
            'seq': torch.tensor([1], device=device, dtype=torch.long),  # any int number
            'imgid': torch.tensor([self.global_step], device=device, dtype=torch.long),
            'global_step': self.global_step
        }

        return dps

    def out_to_tensor(self, out):
        out_tensor = torch.full([20, 14], -1, device='cuda', dtype=torch.float)
        box = out['left'].bbox
        labels = out['left'].get_field('labels')
        scores = out['left'].get_field('scores')
        trackids = out['left'].get_field('trackids')
        box3d = out['left'].get_field('box3d').convert('xyzhwl_ry').bbox_3d
        nobj = box.shape[0]
        out_tensor[:nobj, :4] = box
        out_tensor[:nobj, 4] = labels
        out_tensor[:nobj, 5] = scores
        out_tensor[:nobj, 6] = trackids
        out_tensor[:nobj, 7:] = box3d
        return out_tensor

    def forward(self, inputs):
        """
        :param inputs: 4x3xhxw
        :return:
        """
        dps = self.inputs_to_dps(inputs)
        out, loss_dict = self.model(dps)
        out_tensor = self.out_to_tensor(out)
        return out_tensor


def main():
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
    # os.makedirs(cfg.output_dir, exist_ok=True)
    # Vis3D.default_out_folder = osp.join(cfg.output_dir, 'dbg')
    logger = setup_logger(cfg.output_dir, get_rank(), 'logtest.txt')
    trainer = build_trainer(cfg)
    trainer.resume()

    valid_ds = trainer.valid_dl.dataset
    data0 = valid_ds[0]
    calib = data0['targets']['left'].extra_fields['calib']
    model = trainer.model
    # model.cpu()
    model = FCN_ResNet101(model, calib)
    model.eval()

    # Generate input tensor with random values
    input_tensor = torch.rand(4, 3, 375, 1242)
    input_tensor = input_tensor.cuda()

    # output_tensor = model(input_tensor.cuda())

    # Export torch model to ONNX
    print("Exporting ONNX model {}".format(output_onnx))
    torch.onnx.export(model, input_tensor, output_onnx,
                      # opset_version=12,
                      do_constant_folding=True,
                      input_names=["input"],
                      output_names=["output"],
                      # dynamic_axes={"input": {0: "batch", 2: "height", 3: "width"},
                      #               "output": {0: "batch"}
                      #               },
                      verbose=True)


if __name__ == '__main__':
    main()
