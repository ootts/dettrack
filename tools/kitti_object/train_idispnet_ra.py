import argparse
import os
from warnings import warn

import torch
import torch.multiprocessing
import torch.nn.functional as F
from disprcnn.utils.stereo_utils import end_point_error
from dl_ext import AverageMeter
from fastai.train import fit_one_cycle
from tqdm import tqdm, trange

from disprcnn.utils.fastai_ext import TensorBoardCallback, DistributedSaveModelCallback, LogCallback
from fastai.basic_data import DataBunch
from fastai.basic_train import Learner
from torch import nn
from torch.distributed import get_rank
from torch.utils.data import DataLoader, ConcatDataset
from fastai.distributed import *  # do not delete this line!
# from disprcnn.data.datasets import KITTIRoiDataset
# from disprcnn.data.datasets import KITTIRoiDatasetRA
from disprcnn.modeling.models.psmnet.stackhourglass import PSMNet


class PSMLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, output, y):
        mask = y['mask']
        target = y['disparity']
        # mask = mask.bool() & (target < args.maxdisp) & (target > args.mindisp)
        if isinstance(output, (list, tuple)) and len(output) == 3:
            training = True
        else:
            training = False
        if training:
            output1, output2, output3 = output
            loss1 = (F.smooth_l1_loss(output1, target, reduction='none') * mask.float()).sum()
            loss2 = (F.smooth_l1_loss(output2, target, reduction='none') * mask.float()).sum()
            loss3 = (F.smooth_l1_loss(output3, target, reduction='none') * mask.float()).sum()
            if mask.sum() != 0:
                loss1 = loss1 / mask.sum()
                loss2 = loss2 / mask.sum()
                loss3 = loss3 / mask.sum()
            loss = 0.5 * loss1 + 0.7 * loss2 + loss3
        else:
            if mask.sum() == 0:
                loss = mask.sum() * 0.0
            else:
                loss = ((output - target).abs() * mask.float()).sum() / mask.sum()
        return loss


def evaluate(learner: Learner, dataset):
    if dataset == 'valid':
        ds = learner.data.valid_dl.dataset
    else:
        ds = learner.data.train_dl.dataset
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    # debug
    model = learner.model
    model.eval()
    preds = []
    am = AverageMeter()
    with torch.no_grad():
        for inputs, targets in tqdm(dl):
            inputs = {k: v.cuda() for k, v in inputs.items()}
            output = model(inputs).cpu()
            mask, disparity = targets['mask'], targets['disparity']
            epe = end_point_error(disparity, mask, output)
            am.update(epe, mask.sum().item())
            preds.append(output.cpu())
    print(am.avg)
    preds = torch.cat(preds)
    print('Computing epe.')
    am2 = AverageMeter()
    epes = []
    for i in trange(len(ds)):
        pred = preds[i]
        targets = ds.get_target(i)
        mask, target = targets['mask'], targets['disparity']
        epe = end_point_error(target, mask, pred)
        # epe = rmse(target, mask, pred)
        epes.append(epe)
        am2.update(epe, mask.sum().item())
    print('Average epe', am2.avg)


def main():
    from disprcnn.engine.defaults import setup
    from disprcnn.engine.defaults import default_argument_parser
    from disprcnn.data import make_data_loader
    parser = default_argument_parser()
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    args.config_file = 'configs/idispnet/kitti.yaml'
    cfg = setup(args)

    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
    args.ngpus = num_gpus

    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model = PSMNet(cfg).cuda()
    train_dl = make_data_loader(cfg, is_train=True)
    val_dl = make_data_loader(cfg, is_train=False)

    loss_fn = PSMLoss()

    databunch = DataBunch(train_dl, val_dl, device='cuda')
    learner = Learner(databunch, model, loss_func=loss_fn, model_dir=cfg.output_dir)
    learner.callbacks = [
        DistributedSaveModelCallback(learner, every='epoch' if cfg.solver.save_every else 'improvement'),
        TensorBoardCallback(learner),
        LogCallback(learner, cfg.output_dir)]
    if num_gpus > 1:
        learner.to_distributed(get_rank())
    if cfg.solver.load != '':
        learner.load(cfg.solver.load)
    if args.mode == 'train':
        learner.fit(cfg.solver.num_epochs, cfg.solver.maxlr)
    elif args.mode == 'train_oc':
        fit_one_cycle(learner, cfg.solver.num_epochs, cfg.solver.max_lr)
    elif args.mode == 'eval_train':
        evaluate(learner, 'train')
    elif args.mode == 'eval':
        evaluate(learner, 'valid')
    else:
        raise ValueError('args.mode not supported.')


if __name__ == "__main__":
    main()
