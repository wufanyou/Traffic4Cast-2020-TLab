from utils import *
import argparse
import numpy as np
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
import random
import torch
import os

cudnn.benchmark = True
parser = argparse.ArgumentParser(description="Distributed Training")

parser.add_argument(
    "-c", "--city", default="berlin",
)

parser.add_argument(
    "--config", default="./traffic4cast2020/HRNETV1.yaml", type=str,
)

parser.add_argument(
    "-j",
    "--workers",
    default=8,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 4)",
)
parser.add_argument(
    "--world-size",
    default=-1,
    type=int,
    help="number of nodes for distributed training",
)
parser.add_argument(
    "--rank", default=-1, type=int, help="node rank for distributed training"
)

parser.add_argument(
    "--dist-url", type=str, help="url used to set up distributed training",
)

parser.add_argument(
    "-p",
    "--port",
    default=12345,
    type=int,
    help="port used to set up distributed training",
)

parser.add_argument(
    "--dist-backend", default="nccl", type=str, help="distributed backend"
)
parser.add_argument("--gpu", default=None, type=int, help="GPU id to use.")

parser.add_argument(
    "--nodes_set", type=str, default="", help="select nodes format: node03,node04",
)

parser.add_argument(
    "--ngpus_per_node", type=int, default=3,
)

parser.add_argument(
    "--multiprocessing-distributed",
    action="store_true",
    help="Use multi-processing distributed training to launch "
    "N processes per node, which has N GPUs. This is the "
    "fastest way to use PyTorch for either single node or "
    "multi node data parallel training",
)

parser.add_argument(
    "--no-pin", action="store_false",
)

parser.add_argument(
    "--no-check-name", action="store_false",
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    args = parser.parse_args()
    cfg = get_cfg(args.config, args.no_check_name)
    args.world_size = len(args.nodes_set.split(","))
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = args.ngpus_per_node

    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args, cfg))
    else:
        main_worker(args.gpu, ngpus_per_node, args, cfg)


def get_current_node():
    return os.popen("/bin/hostname").read().split("\n")[0]


def get_group_rank(args):
    group_set = args.nodes_set.split(",")
    group_set = dict(zip(group_set, range(len(group_set))))
    node = get_current_node()
    rank = group_set[node]
    return rank


def main_worker(gpu, ngpus_per_node, args, cfg):
    args.gpu = gpu
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    if args.distributed:
        args.rank = get_group_rank(args)

        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu

        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )

    model = get_model(cfg, args.city.upper())

    if not torch.cuda.is_available():
        print("using CPU, this will be slow")

    elif args.distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            model = nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu], find_unused_parameters=True
            )
        else:
            model.cuda()
            model = nn.parallel.DistributedDataParallel(model)

    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        model = nn.DataParallel(model).cuda()

    train_dataset = get_dataset(cfg, args.city, "train")
    valid_dataset = get_dataset(cfg, args.city, "valid")

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset)
    else:
        train_sampler = None
        valid_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.OPTIM.BATCH_SIZE,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=args.no_pin,
        sampler=train_sampler,
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.OPTIM.BATCH_SIZE,
        shuffle=(valid_sampler is None),
        num_workers=args.workers,
        pin_memory=args.no_pin,
        sampler=valid_sampler,
    )

    optim, scheduler = get_optim(cfg, model, len(train_loader))
    writer = get_writer(cfg, args.city, args.rank, cfg.DIST.VERSION)
    saver = get_saver(cfg, args.city, args.rank, args.distributed)
    train_step = 0
    valid_step = 0

    for epoch in range(1, cfg.OPTIM.MAX_EPOCH + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        train_step = train_model(
            train_step, epoch, train_loader, model, optim, scheduler, writer
        )
        valid_step = valid_model(valid_step, epoch, valid_loader, model, writer)
        saver.save(model, epoch, writer.meter.avg)


if __name__ == "__main__":
    main()
