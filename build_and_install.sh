#!/bin/bash
/bin/rm -r build/ disprcnn.egg-info
python setup.py build develop
#cd disprcnn/modeling/models/pointnet_module/point_rcnn/lib/pointnet2_lib/pointnet2
#/bin/rm -r build/ dist/ pointnet2.egg-info
#python setup.py install
#
#cd ../../utils/iou3d/
#/bin/rm -r build dist iou3d.egg-info
#python setup.py install
#
#cd ../roipool3d/
#/bin/rm -r build dist roipool3d.egg-info
#python setup.py install

cd ../../../../../../../../

chmod +x tools/kitti_object/kitti_evaluation_lib/*

#pip install --user --pre https://storage.googleapis.com/open3d-releases-master/python-wheels/open3d-0.13.0-cp38-cp38-linux_x86_64.whl