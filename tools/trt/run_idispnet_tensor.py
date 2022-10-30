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
    if cfg.trt.convert_to_trt.fp16:
        engine_file = engine_file.replace(".engine", "-fp16.engine")
        
    inferencer = IDispnetInference(engine_file)

    pred_idisp = inferencer.predict_idisp(left_roi_images, right_roi_images)
    inferencer.destory()

    ref = torch.load('tmp/outputs.pth', 'cuda')
    print((pred_idisp - ref).abs().max())


if __name__ == '__main__':
    main()
