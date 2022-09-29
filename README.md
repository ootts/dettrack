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

   Or download data and models from [BaiduYun](**https://pan.baidu.com/s/1pyJ3ul8Kf6HHOvZvoF4jMA),   密码: 1bqq.
   
3. Export environment variables
```bash
export CUDA_HOME=/usr/local/cuda-10.2
export NUMBAPRO_CUDA_DRIVER=/usr/lib/x86_64-linux-gnu/libcuda.so
export NUMBAPRO_LIBDEVICE=/usr/local/cuda-10.2/nvvm/libdevice
export NUMBAPRO_NVVM=/usr/local/cuda-10.2/nvvm/lib64/libnvvm.so
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
python tools/test_net.py -c configs/drcnn/kitti_tracking/pointpillars_112_demo.yaml model.drcnn.mask_mode mask
tensorboard --logdir models/drcnn/kitti_tracking/pointpillars_112_demo/evaltime/kittitrackingstereo_demo/
# Then open localhost:6006 in your browser and you should see a running time curve.
```

