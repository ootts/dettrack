import datetime
import os
import time

import loguru

from disprcnn.trainer.base import BaseTrainer
from disprcnn.trainer.utils import format_time
from disprcnn.utils.comm import synchronize, is_main_process


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
