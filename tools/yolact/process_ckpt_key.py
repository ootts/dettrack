import torch
import os
import os.path as osp


def main():
    src_dir = osp.expanduser("~/PycharmProjects/yolact/weights")
    dst_dir = "models/yolact/official"
    for file in os.listdir(src_dir):
        if file.endswith(".pth"):
            abs_path = osp.join(src_dir, file)
            dst_path = osp.join(dst_dir, file)
            d = torch.load(abs_path, "cpu")
            new_d = {"model." + k: v for k, v in d.items()}
            torch.save(new_d, dst_path)


if __name__ == '__main__':
    main()
