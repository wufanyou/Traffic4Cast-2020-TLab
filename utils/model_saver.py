# Created by fw at 9/28/20
import os
import torch

__ALL__ = ["get_saver"]


class ModelSaver(object):
    def __init__(self, cfg, city, rank=0, distributed=False, save_worst=False):
        self.cfg = cfg
        self.city = city
        self.best_loss = float("inf")
        self.worst_loss = 0
        self.rank = rank
        self.distributed = distributed
        self.save_worst = save_worst
        # os.makedirs(cfg.CHECKPOINT_PATH, exist_ok=True)
        self.filename = os.path.join(
            cfg.CHECKPOINT_PATH,
            cfg.FORMATER.format(version=cfg.VERSION, city=city, epoch="{epoch}"),
        )

    def save(self, model, epoch, loss):
        if self.rank == 0:
            if epoch % self.cfg.STORE_FREC == 0:
                output_name = self.filename.format(epoch=epoch)
                if self.distributed:
                    torch.save(model.module.state_dict(), output_name)
                else:
                    torch.save(model.state_dict(), output_name)

            if loss <= self.best_loss:
                output_name = self.filename.format(epoch="best")
                if self.distributed:
                    torch.save(model.module.state_dict(), output_name)
                else:
                    torch.save(model.state_dict(), output_name)
                self.best_loss = loss

            if self.save_worst:
                if loss >= self.worst_loss:
                    output_name = self.filename.format(epoch="worst")
                    if self.distributed:
                        torch.save(model.module.state_dict(), output_name)
                    else:
                        torch.save(model.state_dict(), output_name)
                    self.worst_loss = loss

    def save_model(self, model, epoch):
        output_name = self.filename.format(epoch=epoch)
        torch.save(model.state_dict(), output_name)

    def load(self, model, epoch=None):
        epoch = "best" if epoch is None else epoch
        output_name = self.filename.format(epoch=epoch)
        model.load_state_dict(torch.load(output_name, map_location="cpu"))
        print(f"load {output_name}")


def get_saver(cfg, city, rank=0, mdp=False):
    cfg = cfg.DIST
    saver = ModelSaver(cfg, city, rank, mdp)
    return saver
