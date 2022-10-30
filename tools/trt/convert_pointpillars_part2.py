import os.path as osp
import os
from disprcnn.config import cfg

from disprcnn.engine.defaults import default_argument_parser
from disprcnn.trainer.build import build_trainer
from disprcnn.utils.comm import get_rank
from disprcnn.utils.logger import setup_logger

import torch
import torch.nn as nn


class PointPillarsPart2Onnx(nn.Module):
    def __init__(self, model):
        super(PointPillarsPart2Onnx, self).__init__()
        self.model = model.pointpillars

    def forward(self, spatial_features):
        return self.model.forward_onnx_part2(spatial_features)


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
    model = PointPillarsPart2Onnx(model)
    model.eval()
    model.cpu()

    spatial_features = torch.load('tmp/spatial_features.pth', 'cpu')

    output_onnx = osp.join(cfg.trt.onnx_path, "pointpillars_part2.onnx")

    print("Exporting ONNX model {}".format(output_onnx))
    torch.onnx.export(model, spatial_features,
                      output_onnx,
                      opset_version=12,
                      do_constant_folding=True,
                      input_names=["input"],
                      output_names=["box_preds", "cls_preds", "dir_cls_preds"],
                      # dynamic_axes={
                      #     "voxels": {0: "batch"},
                      #     "num_points": {0: "batch"},
                      #     "coordinates": {0: "batch"},
                      #     "output": {0: "batch"},
                      # },
                      verbose=False)
    simp_onnx = output_onnx.replace('.onnx', '-simp.onnx')
    os.system(f"/home/linghao/anaconda3/envs/pt110/bin/onnxsim {output_onnx} {simp_onnx}")

    print('to engine')
    engine_path = osp.join(cfg.trt.convert_to_trt.output_path, "pointpillars_part2.engine")
    cmd = f"~/Downloads/TensorRT-8.4.1.5/bin/trtexec --onnx={simp_onnx} --workspace=40960 --tacticSources=-cublasLt,+cublas"
    if cfg.trt.convert_to_trt.fp16:
        cmd = cmd + " --fp16"
        engine_path = engine_path.replace(".engine", "-fp16.engine")
        cmd = cmd + f" --saveEngine={engine_path}"
    # cmd = cmd + " --minShapes=voxels:100x100x4,num_points:100,coordinates:100x4" \
    #             " --optShapes=voxels:1403x100x4,num_points:1403,coordinates:1403x4" \
    #             " --maxShapes=voxels:3000x100x4,num_points:3000,coordinates:3000x4"
    os.system(cmd)


if __name__ == '__main__':
    main()
