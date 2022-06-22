import torch


def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and (lists of annotations, masks)

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list<tensor>, list<tensor>, list<int>) annotations for a given image are stacked
                on 0 dim. The output gt is a tuple of annotations and masks.
    """
    targets = []
    imgs = []
    masks = []
    num_crowds = []
    heights = []
    widths = []
    imgids = []
    indexs = []

    for sample in batch:
        imgs.append(sample['image'])
        targets.append(torch.from_numpy(sample['target']).float())
        masks.append(torch.from_numpy(sample['masks']).float())
        num_crowds.append(int(sample['num_crowds']))
        heights.append(sample['height'])
        widths.append(sample['width'])
        imgids.append(sample['imgid'])
        indexs.append(sample['index'])
    imgs = torch.stack(imgs)
    heights = torch.tensor(heights)
    widths = torch.tensor(widths)
    imgids = torch.tensor(imgids)
    indexs = torch.tensor(indexs)
    dps = {
        'image': imgs,
        'targets': targets,
        'masks': masks,
        'num_crowds': num_crowds,
        'height': heights,
        'width': widths,
        'imgid': imgids,
        'index': indexs
    }
    return dps


class DetectionBatchCollator(object):
    def __call__(self, batch):
        return detection_collate(batch)
