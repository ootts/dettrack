import torch
import os.path as osp
import datetime
import os
import time

import loguru
from termcolor import colored
from torch.utils.data import DataLoader
from tqdm import tqdm

from disprcnn.trainer.base import BaseTrainer
from disprcnn.trainer.utils import format_time, to_cpu, to_cuda
from disprcnn.utils.comm import synchronize, is_main_process, get_rank, get_world_size


class YolactTrackingTrainer(BaseTrainer):
    def fit(self):
        os.makedirs(self.output_dir, exist_ok=True)
        # self.archive_logs()
        num_epochs = self.num_epochs
        if self.cfg.model.yolact_tracking.fix_yolact:
            for param in self.model.yolact.parameters():
                param.requires_grad = False
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
                if self.cfg.test.training_mode:
                    loguru.logger.warning("Running inference with model.train()!!")
                    self.model.train()
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
                # outputs = list(itertools.chain(*outputs))
            os.makedirs(osp.dirname(prediction_path), exist_ok=True)
            if self.cfg.test.save_predictions and get_rank() == 0:
                torch.save(outputs, prediction_path)
        return outputs
