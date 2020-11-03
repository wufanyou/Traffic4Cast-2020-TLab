from apex.optimizers import FusedLAMB, FusedAdam
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import AdamW, Adam, SGD
import math

# https://huggingface.co/transformers/_modules/transformers/optimization.html#get_linear_schedule_with_warmup
def get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, last_epoch=-1
):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0,
    after a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step)
            / float(max(1, num_training_steps - num_warmup_steps)),
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        num_cycles (:obj:`float`, `optional`, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(
            0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_exponent_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    exponent: float = 1 - 2e-3,
    step: int = 10,
    last_epoch: int = -1,
):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        else:
            return exponent ** ((current_step - num_warmup_steps) // step)

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_optim(cfg, model, dataset_iter_num):
    cfg = cfg.OPTIM
    optim_name = cfg.NAME
    optimizer = None
    assert optim_name in ["FusedLAMB", "AdamW", "Adam", "SGD"], "optimizer not allowed"
    parameters = filter(lambda p: p.requires_grad, model.parameters())

    if optim_name == "FusedLAMB":
        optimizer = FusedLAMB(parameters, lr=cfg.INIT_LR, eps=cfg.ADAM_EPSILON)
    if optim_name == "AdamW":
        optimizer = AdamW(parameters, lr=cfg.INIT_LR, eps=cfg.ADAM_EPSILON)
    if optim_name == "Adam":
        optimizer = Adam(parameters, lr=cfg.INIT_LR, eps=cfg.ADAM_EPSILON)
    if optim_name == "SGD":
        optimizer = SGD(parameters, lr=cfg.INIT_LR, momentum=cfg.SGD_MOMENTUM)
    warmup_step = int(cfg.WARM_UP_EPOCH * dataset_iter_num)
    max_step = cfg.MAX_EPOCH * dataset_iter_num

    if cfg.USE_LR_SCHEDULER:
        if cfg.LR_SCHEDULER_TYPE == "get_exponent_schedule_with_warmup":
            scheduler = get_exponent_schedule_with_warmup(
                optimizer, warmup_step, exponent=cfg.EXPONENT
            )
        else:
            scheduler = globals()[cfg.LR_SCHEDULER_TYPE](
                optimizer, warmup_step, max_step
            )
    else:
        scheduler = None

    return optimizer, scheduler
