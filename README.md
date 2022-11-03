# 3D Object Detection and Tracking

## Overview
Given a sequence of stereo images, this project aims to perform 3D object detection and tracking.

The system is based on [Disp R-CNN](https://arxiv.org/pdf/2004.03572.pdf) with the 2D and 3D detectors replaced with [Yolact](https://arxiv.org/pdf/1904.02689.pdf) and [PointPillars](https://arxiv.org/pdf/1812.05784.pdf) for faster running speed.

## Requirements

- Ubuntu 18.04+
- Python 3.7
- Nvidia GPU
- PyTorch 1.10.0
- cudatoolkit==10.2
- boost<=1.71 (recommended)

## Install

```bash
# clone
git clone https://github.com/ootts/dettrack.git
# install conda environment
conda create -n dettrack python=3.7 -y
conda activate dettrack
conda install pytorch=1.10.0 torchvision=0.11.1 cudatoolkit=10.2 -c pytorch
pip install -r requirements.txt
sh build_and_install.sh
```

## Inference example on KITTI
1. Prepare data
   download from [Google drive](https://drive.google.com/file/d/1uokHLQg6CuwqchJiMIvbiWaAAYAPN3qH/view?usp=sharing) and extract into PROJECTROOT/data

2. Prepare models.
   download from [Google drive](https://drive.google.com/file/d/15sJ4msyCSwnYBgRb8eEFGzkRmK9QnObV/view?usp=sharing) and extract into PROJECTROOT/models.

   Note that only **cars** are supported by now.

   Or download data and models from [BaiduYun](https://pan.baidu.com/s/1pyJ3ul8Kf6HHOvZvoF4jMA),   密码: 1bqq.
   
3. Export environment variables
```bash
export CUDA_HOME=/usr/local/cuda-10.2
export PATH=/usr/local/cuda-10.2/bin:$PATH
```

1. Run inference and visualization.

```bash
cd PROJECTROOT
wis3d --host localhost --vis_dir dbg --port 19090
# Then open http://localhost:19090/?tab=3d&sequence=drcnn_vis_final_result in your browser.
python tools/test_net.py -c configs/drcnn/kitti_tracking/pointpillars_112_demo.yaml dbg True
```

4. Run inference and evaluate running time.

```bash
python tools/test_net.py -c configs/drcnn/kitti_tracking/pointpillars_112_demo.yaml model.drcnn.mask_mode mask evaltime True
tensorboard --logdir models/drcnn/kitti_tracking/pointpillars_112_demo/evaltime/kittitrackingstereo_demo/
# Then open localhost:6006 in your browser and you should see a running time curve.
```



## Inference with TensorRT

### Requirements

GPU sm>=70, such as GTX 2080.

Ubuntu 16.04 or later

cuda 10.2

cudnn 8.4.1 [install](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html)

tensorRT [install](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-tar)

onnxeruntime, onnxsim (install using pip)

### Run

1. Convert pytorch model to xxx.onnx.
   Since tensorRT can not deal with the entire model, we select some parts from the model and convert them one by one.

   ```bash
   sh tools/trt/convert.sh # see this script for detailed commands.
   ```

2. Simplify xxx.onnx.
   Use "onnxsim" to simplify xxx.onnx. This step has been included in Step1.

3. Convert xxx.onnx to xxx.engine.
   Use "trtexec" to perform the conversion. This step has been included in Step1.

4. Load xxx.engine for inference.
   After converting all models, run "tools/trt/run_all.py" to load engines and perform inference. The running time will be evaluated.

5. NOTE: See configs/drcnn/kitti_tracking/pointpillars_112_600x300_demo.yaml and modify "trt.convert_to_trt.fp16" to enable or disable fp16. Enabling fp16 usually further speeds up the running speed, while losing a little numerical precision.

