import os.path as osp
import os
from disprcnn.config import cfg

from disprcnn.engine.defaults import default_argument_parser
from disprcnn.trainer.build import build_trainer

import torch
import torch.nn as nn


class YolactOnnx(nn.Module):
    def __init__(self, model):
        super(YolactOnnx, self).__init__()
        self.model = model.yolact_tracking.yolact

    def forward(self, inputs):
        """
        :param inputs: 2x3x300x600
        :return:
        """
        return self.model.forward_onnx(inputs)


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
    model = YolactOnnx(model)
    model.eval()
    model.cpu()

    # Generate input tensor with random values
    input_tensor = torch.rand(2, 3, 300, 600).float()

    # Export torch model to ONNX
    output_onnx = osp.join(cfg.trt.onnx_path, "yolact.onnx")
    print("Exporting ONNX model {}".format(output_onnx))
    torch.onnx.export(model, input_tensor, output_onnx,
                      opset_version=12,
                      do_constant_folding=True,
                      input_names=["input"],
                      output_names=['loc', 'conf', 'mask', 'priors', 'proto', 'feat_out'],
                      # dynamic_axes={"input": {0: "batch", 2: "height", 3: "width"},
                      #               "output": {0: "batch"}
                      #               },
                      verbose=False)

    simp_onnx = output_onnx.replace('.onnx', '-simp.onnx')
    os.system(f"/home/linghao/anaconda3/envs/pt110/bin/onnxsim {output_onnx} {simp_onnx}")

    print('to engine')
    engine_path = osp.join(cfg.trt.convert_to_trt.output_path, "yolact.engine")
    cmd = f"~/Downloads/TensorRT-8.4.1.5/bin/trtexec --onnx={simp_onnx} --workspace=40960 --saveEngine={engine_path}  --tacticSources=-cublasLt,+cublas"
    if cfg.trt.convert_to_trt.fp16:
        cmd = cmd + " --fp16"
    os.system(cmd)


if __name__ == '__main__':
    main()
