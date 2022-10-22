import torch


def out_to_tensor(out):
    out_tensor = torch.full([20, 14], -1, device='cuda', dtype=torch.float)
    box = out['left'].bbox
    labels = out['left'].get_field('labels')
    scores = out['left'].get_field('scores')
    trackids = out['left'].get_field('trackids')
    box3d = out['left'].get_field('box3d').convert('xyzhwl_ry').bbox_3d
    nobj = box.shape[0]
    out_tensor[:nobj, :4] = box
    out_tensor[:nobj, 4] = labels
    out_tensor[:nobj, 5] = scores
    out_tensor[:nobj, 6] = trackids
    out_tensor[:nobj, 7:] = box3d
    return out_tensor


def main():
    outputs = torch.load('tmp/outputs.pth')
    ot = out_to_tensor(outputs)
    print()


if __name__ == '__main__':
    main()
