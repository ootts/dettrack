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
from disprcnn.structures.bounding_box_3d import Box3DList
from disprcnn.structures.boxlist_ops import boxlist_iou
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

output_onnx = "tmp/idispnet.onnx"


# FC-ResNet101 pretrained model from torch-hub extended with argmax layer
class IDispnetOnnx(nn.Module):
    def __init__(self, model):
        super(IDispnetOnnx, self).__init__()
        self.model = model.idispnet

    def forward(self, inputs):
        """
        :param inputs: 2x3x112x112
        :return:
        """
        # N = inputs.shape[0] // 2
        left, right = inputs[:1], inputs[1:]
        pred_outs = self.model.forward_onnx(left, right)
        return pred_outs


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
    logger = setup_logger(cfg.output_dir, get_rank(), 'logtest.txt')
    trainer = build_trainer(cfg)
    trainer.resume()

    valid_ds = trainer.valid_dl.dataset
    data0 = valid_ds[0]
    calib = data0['targets']['left'].extra_fields['calib']
    model = trainer.model
    model = IDispnetOnnx(model)
    model.eval()
    model.cuda()

    input_tensor = torch.rand(2, 3, 112, 112).float().cuda()

    print("Exporting ONNX model {}".format(output_onnx))
    torch.onnx.export(model, input_tensor, output_onnx,
                      opset_version=12,
                      do_constant_folding=True,
                      input_names=["input"],
                      output_names=["output"],
                      # dynamic_axes={"input": {0: "batch"},
                      #               "output": {0: "batch"}
                      #               },
                      verbose=False)


if __name__ == '__main__':
    main()
