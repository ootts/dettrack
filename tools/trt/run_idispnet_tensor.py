import torch
import os.path as osp
import pycuda.driver as cuda

import pycuda.autoinit
from disprcnn.trt.idispnet_inference import IDispnetInference


def main():
    from disprcnn.engine.defaults import setup
    from disprcnn.engine.defaults import default_argument_parser
    from disprcnn.data import make_data_loader
    parser = default_argument_parser()
    args = parser.parse_args()
    args.config_file = 'configs/drcnn/kitti_tracking/pointpillars_112_600x300_demo.yaml'
    cfg = setup(args)

    left_roi_images, right_roi_images = torch.load('tmp/left_right_roi_images.pth', 'cuda')

    engine_file = osp.join(cfg.trt.convert_to_trt.output_path, "idispnet.engine")

    inferencer = IDispnetInference(engine_file)

    inferencer.infer(left_roi_images, right_roi_images)
    inferencer.destory()

    ref = torch.load('tmp/outputs.pth', 'cuda')
    print((inferencer.cuda_outputs['output'][:6] - ref).abs().max())


if __name__ == '__main__':
    main()
