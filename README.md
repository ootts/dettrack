# 3D Object Detection and Tracking from Stereo Images

## Overview
Given a sequence of stereo images, this project aims to perform 3D object detection and tracking.

The system is based on [Disp R-CNN](https://arxiv.org/pdf/2004.03572.pdf) with the 2D and 3D detectors replaced with [Yolact](https://arxiv.org/pdf/1904.02689.pdf) and [PointPillars](https://arxiv.org/pdf/1812.05784.pdf) for faster running speed.

## Requirements

- Ubuntu 16.04+
- Python 3.7
- Nvidia GPU
- PyTorch 1.10.0
- boost 1.65.1 (recommended)

## Install

```bash
# clone
git clone https://github.com/ootts/dettrack.git
# install conda environment
conda env create -f environment.yaml
conda activate dettrack
sh build_and_install.sh
```

## Inference example on KITTI
1. Prepare data

2. Download trained models.

3. Run inference and visualization.
```bash
cd PROJECTROOT
wis3d --host localhost --vis_dir dbg
# Then open localhost:19090 in your browser.
python tools/test_net.py -c configs/drcnn/kitti_tracking/pointpillars_112_demo.yaml dbg True
```

4. Run inference and evaluate running time.

```bash
python tools/test_net.py -c configs/drcnn/kitti_tracking/pointpillars_112_demo.yaml model.drcnn.mask_mode mask
tensorboard --logdir models/drcnn/kitti_tracking/pointpillars_112_demo/evaltime/kittitrackingstereo_demo/
# Then open localhost:6006 in your browser and you should see a running time curve.
```

