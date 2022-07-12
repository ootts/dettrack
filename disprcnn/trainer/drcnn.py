import itertools

import torch
import os.path as osp
import datetime
import os
import time

import loguru
from dl_ext.pytorch_ext import OneCycleScheduler, reduce_loss
from termcolor import colored
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
from torch.utils.data import DataLoader
from tqdm import tqdm

from disprcnn.data.samplers.ordered_distributed_sampler import OrderedDistributedSampler
from disprcnn.solver.lr_scheduler import WarmupMultiStepLR
from disprcnn.trainer.base import BaseTrainer
from disprcnn.trainer.utils import format_time, to_cpu, to_cuda
from disprcnn.utils.averagemeter import AverageMeter
from disprcnn.utils.comm import synchronize, is_main_process, get_rank, get_world_size, all_gather


class DRCNNTrainer(BaseTrainer):
    def fit(self):
        os.makedirs(self.output_dir, exist_ok=True)
        # self.archive_logs()
        num_epochs = self.num_epochs
        if self.cfg.model.drcnn.fix_yolact:
            for param in self.model.yolact.parameters():
                param.requires_grad = False
            self.model.yolact.eval()
        if self.cfg.model.drcnn.fix_idispnet:
            for param in self.model.idispnet.parameters():
                param.requires_grad = False
            self.model.idispnet.eval()
        begin = time.time()
        for epoch in range(self.begin_epoch, num_epochs):
            epoch_begin = time.time()
            self.train(epoch)
            synchronize()
            if self.save_every is False and epoch % self.cfg.solver.val_freq == 0:
                self.val_loss = self.val(epoch)
                synchronize()
            if is_main_process():
                # synchronize()
                self.epoch_time_am.update(time.time() - epoch_begin)
                eta = (num_epochs - epoch - 1) * self.epoch_time_am.avg
                finish_time = datetime.datetime.now() + datetime.timedelta(seconds=int(eta))
                loguru.logger.info(
                    f"ETA: {format_time(eta)}, finish time: {finish_time.strftime('%m-%d %H:%M')}")
            if (1 + epoch) % self.cfg.solver.save_freq == 0 and epoch != self.begin_epoch:
                self.try_to_save(epoch, 'epoch')

            synchronize()
        if is_main_process():
            loguru.logger.info('Training finished. Total time %s' % (format_time(time.time() - begin)))

    @torch.no_grad()
    def get_preds(self):
        prediction_path = osp.join(self.cfg.output_dir, 'inference', self.cfg.datasets.test, 'predictions.pth')
        if not self.cfg.test.force_recompute and osp.exists(prediction_path):
            loguru.logger.info(colored(f'predictions found at {prediction_path}, skip recomputing.', 'red'))
            outputs = torch.load(prediction_path)
        else:
            if get_world_size() > 1:
                outputs = self.get_preds_dist()
            else:
                self.model.eval()
                ordered_valid_dl = DataLoader(self.valid_dl.dataset, self.valid_dl.batch_size, shuffle=False,
                                              sampler=None, num_workers=self.valid_dl.num_workers,
                                              collate_fn=self.valid_dl.collate_fn, pin_memory=self.valid_dl.pin_memory,
                                              timeout=self.valid_dl.timeout,
                                              worker_init_fn=self.valid_dl.worker_init_fn)
                bar = tqdm(ordered_valid_dl)
                outputs = []
                for i, batch in enumerate(bar):
                    batch = to_cuda(batch)
                    batch['global_step'] = i
                    output, loss_dict = self.model(batch)
                    output = to_cpu(output)
                    outputs.append(output)
            os.makedirs(osp.dirname(prediction_path), exist_ok=True)
            if self.cfg.test.save_predictions and get_rank() == 0:
                torch.save(outputs, prediction_path)
        return outputs

    @torch.no_grad()
    def get_preds_dist(self):
        self.model.eval()
        valid_sampler = OrderedDistributedSampler(self.valid_dl.dataset, get_world_size(), rank=get_rank())
        ordered_dist_valid_dl = DataLoader(self.valid_dl.dataset, self.valid_dl.batch_size, shuffle=False,
                                           sampler=valid_sampler, num_workers=self.valid_dl.num_workers,
                                           collate_fn=self.valid_dl.collate_fn, pin_memory=self.valid_dl.pin_memory,
                                           timeout=self.valid_dl.timeout,
                                           worker_init_fn=self.valid_dl.worker_init_fn)
        bar = tqdm(ordered_dist_valid_dl) if is_main_process() else ordered_dist_valid_dl
        outputs = []
        for i, batch in enumerate(bar):
            batch = to_cuda(batch)
            batch['global_step'] = i
            output, loss_dict = self.model(batch)
            output = to_cpu(output)
            outputs.append(output)
        torch.cuda.empty_cache()
        all_outputs = all_gather(outputs)
        if not is_main_process():
            return
        all_outputs = list(itertools.chain(*all_outputs))
        all_outputs = all_outputs[:len(self.valid_dl.dataset)]
        return all_outputs

    def train(self, epoch):
        loss_meter = AverageMeter()
        metric_ams = {}
        loss_ams = {}
        # for metric in self.metric_functions.keys():
        #     metric_ams[metric] = AverageMeter()
        self.model.train()
        if self.cfg.model.drcnn.fix_yolact:
            for param in self.model.yolact.parameters():
                param.requires_grad = False
            self.model.yolact.eval()
        if self.cfg.model.drcnn.fix_idispnet:
            for param in self.model.idispnet.parameters():
                param.requires_grad = False
            self.model.idispnet.eval()
        bar = tqdm(self.train_dl, leave=False) if is_main_process() else self.train_dl
        begin = time.time()
        for batch in bar:
            # for i in range(len(batch['image'])):
            #     self.tb_writer.add_image("input"+str(i), batch['image'][i])
            self.optimizer.zero_grad()
            batch = to_cuda(batch)
            batch['global_step'] = self.global_steps
            output, loss_dict = self.model(batch)
            loss = sum(v for k, v in loss_dict.items())
            loss.backward()
            if self.cfg.solver.do_grad_clip:
                if self.cfg.solver.grad_clip_type == 'norm':
                    clip_grad_norm_(self.model.parameters(), self.cfg.solver.grad_clip)
                else:
                    clip_grad_value_(self.model.parameters(), self.cfg.solver.grad_clip)
            self.optimizer.step()
            if self.scheduler is not None and isinstance(self.scheduler, (OneCycleScheduler, WarmupMultiStepLR)):
                self.scheduler.step()
            # record and plot loss and metrics
            reduced_loss = reduce_loss(loss)
            metrics = {}
            if 'metrics' in output:
                for k, v in output['metrics'].items():
                    reduced_s = reduce_loss(v)
                    metrics[k] = reduced_s
            for k, v in loss_dict.items():
                if k not in loss_ams.keys():
                    loss_ams[k] = AverageMeter()
                loss_ams[k].update(v.item())
            if is_main_process():
                loss_meter.update(reduced_loss.item())
                lr = self.optimizer.param_groups[0]['lr']
                self.tb_writer.add_scalar('train/loss', reduced_loss.item(), self.global_steps)
                self.tb_writer.add_scalar('train/lr', lr, self.global_steps)
                for k, v in loss_dict.items():
                    self.tb_writer.add_scalar(f'train/loss/{k}', v.item(), self.global_steps)
                    # self.tb_writer.add_scalar(f'train/loss/smooth_{k}', loss_ams[k].avg, self.global_steps)
                bar_vals = {'epoch': epoch, 'phase': 'train', 'loss': loss_meter.avg}
                for k, v in metrics.items():
                    if k not in metric_ams.keys():
                        metric_ams[k] = AverageMeter()
                    metric_ams[k].update(v.item())
                    self.tb_writer.add_scalar(f'train/{k}', v.item(), self.global_steps)
                    # self.tb_writer.add_scalar(f'train/smooth_{k}', metric_ams[k].avg, self.global_steps)
                    bar_vals[k] = metric_ams[k].avg
                bar.set_postfix(bar_vals)
            self.global_steps += 1
            if self.global_steps % self.save_freq == 0:
                self.try_to_save(epoch, 'iteration')
        torch.cuda.synchronize()
        epoch_time = format_time(time.time() - begin)
        if is_main_process():
            metric_msgs = ['epoch %d, train, loss %.4f, time %s' % (
                epoch, loss_meter.avg, epoch_time)]
            for metric, v in metric_ams.items():
                metric_msgs.append('%s %.4f' % (metric, v.avg))
            s = ', '.join(metric_msgs)
            loguru.logger.info(s)
        if self.scheduler is not None and not isinstance(self.scheduler, (OneCycleScheduler, WarmupMultiStepLR)):
            self.scheduler.step()
