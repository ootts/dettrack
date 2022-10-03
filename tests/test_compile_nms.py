from pathlib import Path

from disprcnn.utils.ppp_utils.buildtools.pybind11_build import load_pb11

try:
    from disprcnn.utils.ppp_utils.non_max_suppression.nms import (
        non_max_suppression_cpu, rotate_non_max_suppression_cpu)
except:
    current_dir = Path(__file__).resolve().parents[0]
    load_pb11(
        ["../cc/nms/nms_kernel.cu.cc", "../cc/nms/nms.cc"],
        current_dir / "nms.so",
        current_dir,
        # includes=["/raid/linghao/project_data/dettrack/external/boost_1_65_1/boost"],
        cuda=True)
    from disprcnn.utils.ppp_utils.non_max_suppression.nms import (
        non_max_suppression_cpu, rotate_non_max_suppression_cpu)
