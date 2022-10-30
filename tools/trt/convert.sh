export CUDA_VISIBLE_DEIVCES="3"
python tools/trt/convert_yolact.py
python tools/trt/convert_yolact_tracking.py
python tools/trt/convert_idispnet.py
python tools/trt/convert_pointpillars.py
python tools/trt/convert_pointpillars_part2.py