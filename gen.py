# Created by fw at 10/1/20
from utils import *
import torch
import argparse
import os
from collections import defaultdict
import numpy as np
import json
import h5py
import pandas as pd
import glob
from datetime import datetime
from tqdm import tqdm
import multiprocessing

result = {}


def write_data_to_h5(data, filename):
    """
    write data in gzipped h5 format
    """
    f = h5py.File(filename, "w", libver="latest")
    dset = f.create_dataset(
        "array", shape=(data.shape), data=data, compression="gzip", compression_opts=9
    )
    f.close()


def phase_index(index):
    dayofyear = index // 288 + 1
    date = pd.to_datetime("2019-01-01") + (
        pd.to_datetime("2019-01-02") - pd.to_datetime("2019-01-01")
    ) * (dayofyear - 1)
    idx = index % 288
    return str(date.date()), idx


def func(k, v, path):
    d = defaultdict(lambda: {})
    for index, value in result.items():
        date, idx = phase_index(index)
        value = value.reshape(6, 8, 495, 436)
        value = np.moveaxis(value, 1, -1)
        d[date][idx] = value

    array = np.stack([d[k][i] for i in v])
    filename = os.path.join(path, f"{k}_test.h5")
    write_data_to_h5(array, filename)
    return 1


parser = argparse.ArgumentParser()
parser.add_argument(
    "-t",
    "--test-slots",
    default="./traffic4cast2020/NeurIPS2020-traffic4cast/core-competition/util/test_slots.json",
    type=str,
)
parser.add_argument(
    "-c", "--city", default="BERLIN", type=str,
)

parser.add_argument(
    "--config",
    default="./traffic4cast2020/config/v1-hernet-base-lr-001.yaml",
    type=str,
)

parser.add_argument(
    "-p", "--path", default="./traffic4cast2020/submission", type=str,
)

parser.add_argument(
    "--tag", default="", type=str,
)

if __name__ == "__main__":
    now = datetime.now()
    args = parser.parse_args()
    if args.city == "0":
        args.city = "berlin"
    elif args.city == "1":
        args.city = "istanbul"
    elif args.city == "2":
        args.city = "moscow"
    args.city = args.city.upper()
    cfg = get_cfg(args.config, False)
    model = get_model(cfg)
    args.tag = None if args.tag == "" else args.tag
    saver = get_saver(cfg, args.city.lower(), args.tag)
    saver.load(model)
    dataset = get_dataset(cfg, args.city, "test")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=8)
    model = model.eval().cuda()

    print(f"[{args.city}] process output:")
    result = test_model(dataloader, model)

    print(f"[{args.city}] process h5:")
    with open(args.test_slots) as f:
        test_json = json.load(f)
    test_json = {list(d.keys())[0]: list(d.values())[0] for d in test_json}

    path = os.path.join(
        args.path, f"{str(now.date())[5:]}-{str(now.hour).zfill(2)}", args.city
    )
    os.makedirs(path, exist_ok=True)
    pool = multiprocessing.Pool(processes=40)
    for k, v in test_json.items():
        pool.apply_async(func, (k, v, path,))
    pool.close()
    pool.join()
