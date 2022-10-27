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

output_onnx = "tmp/disprcnn.onnx"


# FC-ResNet101 pretrained model from torch-hub extended with argmax layer
class YolactOnnx(nn.Module):
    def __init__(self, model):
        super(YolactOnnx, self).__init__()
        self.model = model.yolact_tracking.yolact

    def forward(self, inputs):
        """
        :param inputs: 1x3x300x600
        :return:
        """
        # dps = self.inputs_to_dps(inputs)
        pred_outs, outs = self.model.forward_onnx(inputs)
        return pred_outs, outs


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

    torch_triu = torch.triu

    def triu_onnx(x, diagonal=0, out=None):
        # if x.ndim == 3:
        #     assert x.shape[0] == 1
        #     x = x[0]
        assert out is None
        assert len(x.shape) == 3 and x.size(1) == x.size(2)
        x = x[0]
        template = torch_triu(torch.ones((1024, 1024), dtype=torch.uint8, device=x.device),
                              diagonal)  # 1024 is max sequence length
        mask = template[:x.size(0), :x.size(1)]
        return torch.where(mask.bool(), x, torch.zeros_like(x))[None]

    torch.triu = triu_onnx
    # torch.onnx.export(...)  # export your model here
    # torch.triu = torch_triu

    valid_ds = trainer.valid_dl.dataset
    data0 = valid_ds[0]
    calib = data0['targets']['left'].extra_fields['calib']
    model = trainer.model
    model = YolactOnnx(model)
    model.eval()

    # Generate input tensor with random values
    input_tensor = torch.rand(4, 3, 375, 1242)
    input_tensor = input_tensor.cuda()

    dps = torch.load('tmp/dps.pth')
    input_tensor[0, :, :, :] = dps['original_images']['left'][0].permute(2, 0, 1)
    input_tensor[1, :, :, :] = dps['original_images']['right'][0].permute(2, 0, 1)
    input_tensor[2, :, :300, :600] = dps['images']['left'][0]
    input_tensor[3, :, :300, :600] = dps['images']['right'][0]

    # output_tensor = model(input_tensor)

    # Export torch model to ONNX
    # torch.jit.script(model, input_tensor)
    print("Exporting ONNX model {}".format(output_onnx))
    torch.onnx.export(model, input_tensor, output_onnx,
                      opset_version=12,
                      do_constant_folding=False,
                      input_names=["input"],
                      output_names=["output"],
                      # dynamic_axes={"input": {0: "batch", 2: "height", 3: "width"},
                      #               "output": {0: "batch"}
                      #               },
                      verbose=False)


if __name__ == '__main__':
    main()
