import os.path as osp
import os
import sys

from disprcnn.config import cfg
from disprcnn.engine.defaults import default_argument_parser
from disprcnn.trainer.build import build_trainer
from disprcnn.utils.comm import get_rank
from disprcnn.utils.logger import setup_logger

import torch
import torch.nn as nn


class YolactTrackingHeadOnnx(nn.Module):
    def __init__(self, model):
        super(YolactTrackingHeadOnnx, self).__init__()
        self.model = model.yolact_tracking.track_head

    def forward(self, x, ref_x):
        """
        :param x: M,C,7,7
        :param ref_x: N,C,7,7
        :return:
        """
        return self.model.forward_onnx(x, ref_x)


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
    # logger = setup_logger(cfg.output_dir, get_rank(), 'logtest.txt')
    trainer = build_trainer(cfg)
    trainer.resume()

    valid_ds = trainer.valid_dl.dataset
    data0 = valid_ds[0]
    calib = data0['targets']['left'].extra_fields['calib']
    model = trainer.model
    model = YolactTrackingHeadOnnx(model)
    model.eval()
    model.cpu()

    # Generate input tensor with random values
    x = torch.rand(7, 256, 7, 7)
    ref_x = torch.rand(8, 256, 7, 7)

    # Export torch model to ONNX
    output_onnx = osp.join(cfg.trt.onnx_path, "yolact_tracking_head.onnx")
    print("Exporting ONNX model {}".format(output_onnx))
    torch.onnx.export(model, (x, ref_x), output_onnx,
                      opset_version=12,
                      do_constant_folding=True,
                      input_names=["x", "ref_x"],
                      output_names=["output"],
                      dynamic_axes={
                          "x": {0: "batch1"},
                          "ref_x": {0: "batch2"},
                          "output": {0: "height", 1: "width"}
                      },
                      verbose=False)
    simp_onnx = output_onnx.replace('.onnx', '-simp.onnx')
    onnxsim_path = sys.executable.replace("/bin/python", "/bin/onnxsim")
    os.system(f"{onnxsim_path} {output_onnx} {simp_onnx}")

    print('to engine')
    engine_path = osp.join(cfg.trt.convert_to_trt.output_path, "yolact_tracking_head.engine")
    trtexec_path = osp.expanduser(cfg.trt.convert_to_trt.trtexec_path)
    cmd = f"{trtexec_path} --onnx={simp_onnx} --workspace=40960 --tacticSources=-cublasLt,+cublas"
    if cfg.trt.convert_to_trt.fp16:
        cmd = cmd + " --fp16"
        engine_path = engine_path.replace(".engine", "-fp16.engine")
    cmd = cmd + f" --saveEngine={engine_path}"
    cmd = cmd + " --minShapes=x:1x256x7x7,ref_x:1x256x7x7" \
                " --optShapes=x:10x256x7x7,ref_x:10x256x7x7" \
                " --maxShapes=x:200x256x7x7,ref_x:200x256x7x7"
    os.system(cmd)


if __name__ == '__main__':
    main()
