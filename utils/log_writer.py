import os
from torch.utils.tensorboard import SummaryWriter
import sys
import datetime


def get_current_node():
    return os.popen("/bin/hostname").read().split("\n")[0]


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name=None, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

    @property
    def avg(self):
        return self.sum / self.count


class LogWriter(object):
    def __init__(self, cfg, city, rank=0, log_name=None):
        self.cfg = cfg
        self.city = city.upper()
        self.rank = rank
        self.node = get_current_node()
        self.epoch = 0
        if self.rank == 0:
            log_name = (
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
                if log_name is None
                else log_name
            )
            path = os.path.join(cfg.DIR, log_name)
            self.writer = SummaryWriter(path)
        else:
            self.writer = None

        self.log_name = log_name

    def update(self, loss, n, step):
        self.meter.update(loss, n)
        if (self.rank == 0) and (step % self.cfg.STEP == 0):
            self.writer.add_scalar(f"{self.city}/{self.name}/step", loss, step)
            print(
                f"[{self.node}] [{self.epoch}] [{self.city}] [{step}] loss:{loss:.7f}"
            )

    def write(self):
        if self.rank == 0:
            loss = self.meter.avg
            self.writer.add_scalar(f"{self.city}/{self.name}/epoch", loss, self.epoch)

    def reset(self, name, epoch):
        self.epoch = epoch
        self.name = name
        self.meter = AverageMeter()


def get_writer(cfg, city="berlin", rank=0, log_name=None):
    cfg = cfg.LOG
    return LogWriter(cfg, city, rank, log_name)
