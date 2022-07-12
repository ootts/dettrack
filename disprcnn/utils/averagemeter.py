class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, ignore_first=0, enable=True):
        super().__init__()
        self.enable = enable
        self.reset()
        self.ignore_first = ignore_first

    def reset(self):
        if self.enable:
            self.val = 0
            self.avg = 0
            self.sum = 0
            self.count = 0

    def update(self, val, n=1):
        if self.enable:
            if self.ignore_first > 0:
                self.ignore_first -= 1
            else:
                assert isinstance(n, int) and n >= 0
                self.val = val
                self.sum += val * n
                self.count += n
                if self.count != 0:
                    self.avg = self.sum / self.count
