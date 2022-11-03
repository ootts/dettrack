import os.path as osp
import os
from disprcnn.config import cfg

from disprcnn.engine.defaults import default_argument_parser
from disprcnn.trainer.build import build_trainer
from disprcnn.utils.comm import get_rank
from disprcnn.utils.logger import setup_logger

import torch
import torch.nn as nn


class IDispnetPart3Onnx(nn.Module):
    def __init__(self, model):
        super(IDispnetPart3Onnx, self).__init__()
        self.model = model.idispnet

    def forward(self, refimg_fea, targetimg_fea, neg_cost, pos_cost):
        """
        :param inputs: 2x3x112x112
        :return:
        """
        return self.model.forward_onnx_part3(refimg_fea, targetimg_fea, neg_cost, pos_cost)


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
    model = IDispnetPart3Onnx(model)
    model.eval()
    model.cuda()
    refimg_fea = torch.rand(20, 32, 28, 28).float().cuda()
    targetimg_fea = torch.rand(20, 32, 28, 28).float().cuda()
    neg_cost = torch.rand(20, 64, 6, 28, 28).float().cuda()
    pos_cost = torch.rand(20, 64, 5, 28, 28).float().cuda()

    # Export torch model to ONNX
    output_onnx = osp.join(cfg.trt.onnx_path, "idispnet_part3.onnx")
    print("Exporting ONNX model {}".format(output_onnx))
    torch.onnx.export(model, (refimg_fea, targetimg_fea, neg_cost, pos_cost),
                      output_onnx,
                      opset_version=12,
                      do_constant_folding=True,
                      input_names=["refimg_fea", "targetimg_fea", "neg_cost", "pos_cost"],
                      output_names=["output"],
                      dynamic_axes={"refimg_fea": {0: "batch"},
                                    "targetimg_fea": {0: "batch"},
                                    "neg_cost": {0: "batch"},
                                    "pos_cost": {0: "batch"},
                                    },
                      verbose=False)

    simp_onnx = output_onnx.replace('.onnx', '-simp.onnx')
    os.system(f"/home/linghao/anaconda3/envs/pt110/bin/onnxsim {output_onnx} {simp_onnx}")

    print('to engine')
    engine_path = osp.join(cfg.trt.convert_to_trt.output_path, "idispnet_part3.engine")
    cmd = f"~/Downloads/TensorRT-8.4.1.5/bin/trtexec --onnx={output_onnx} --workspace=40960 --saveEngine={engine_path}  --tacticSources=-cublasLt,+cublas"
    if cfg.trt.convert_to_trt.fp16:
        cmd = cmd + " --fp16"
    cmd = cmd + " --minShapes=refimg_fea:1x32x28x28,targetimg_fea:1x32x28x28,neg_cost:1x64x6x28x28,pos_cost:1x64x5x28x28" \
                " --optShapes=refimg_fea:4x32x28x28,targetimg_fea:4x32x28x28,neg_cost:4x64x6x28x28,pos_cost:4x64x5x28x28" \
                " --maxShapes=refimg_fea:20x32x28x28,targetimg_fea:20x32x28x28,neg_cost:20x64x6x28x28,pos_cost:20x64x5x28x28"
    os.system(cmd)


if __name__ == '__main__':
    main()