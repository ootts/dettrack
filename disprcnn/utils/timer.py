import datetime
import time

import colorama
import torch


# from dl_ext import AverageMeter


# class Timer(object):
#     def __init__(self, ignore_first_n=0, enable=True):
#         self.reset(ignore_first_n)
#         self.enable = enable
#
#     @property
#     def average_time(self):
#         if not self.enable:
#             return 0.0
#         return self.total_time / self.calls if self.calls > 0 else 0.0
#
#     def tic(self, synchronize=True):
#         # using time.time instead of time.clock because time time.clock
#         # does not normalize for multithreading
#         if self.enable:
#             if synchronize:
#                 torch.cuda.synchronize()
#             self.start_time = time.perf_counter()
#
#     def toc(self, average=True, synchronize=True):
#         if self.enable:
#             if synchronize:
#                 torch.cuda.synchronize()
#             self.add(time.perf_counter() - self.start_time)
#             if average:
#                 return self.average_time
#             else:
#                 return self.diff
#
#     def add(self, time_diff):
#         if self.ignore_first_n > 0:
#             self.ignore_first_n -= 1
#         else:
#             self.calls += 1
#             self.diff = time_diff
#             self.total_time += self.diff
#
#     def reset(self, ignore_first_n):
#         self.total_time = 0.0
#         self.calls = 0
#         self.start_time = 0.0
#         self.diff = 0.0
#         self.ignore_first_n = ignore_first_n
#
#     def avg_time_str(self):
#         time_str = str(datetime.timedelta(seconds=self.average_time))
#         return time_str


def get_time_str(time_diff):
    time_str = str(datetime.timedelta(seconds=time_diff))
    return time_str


class EvalTime(object):
    def __init__(self, disable=False, do_print=True):
        self.last_time = None
        self.disable = disable
        self.do_print = do_print
        # self.infos = {}

    def __call__(self, info):
        if not self.disable:
            torch.cuda.synchronize()
            t = time.perf_counter()
            if self.last_time is None:
                self.last_time = t
                if self.do_print:
                    print("{}info : {}{} : %f".format(colorama.Fore.CYAN, info, colorama.Style.RESET_ALL) % t)
            else:
                interval = (t - self.last_time) * 1000
                # if info not in self.infos:
                #     self.infos[info] = AverageMeter()
                # self.infos[info].update(interval)
                if self.do_print:
                    print(
                        "{}info : {}{}".format(colorama.Fore.CYAN, info,
                                               colorama.Style.RESET_ALL) + ' : % f, {}interval{} : % f'.format(
                            colorama.Fore.RED,
                            colorama.Style.RESET_ALL) % (t, interval))

                self.last_time = t
                return interval
