# %load process_lmdb.py
import h5py
import numpy as np
import lmdb
import io
import glob
import pandas as pd
import argparse
import json
import os
from tqdm import tqdm
parser = argparse.ArgumentParser()
parser.add_argument(
    "-c",
    "--city",
    default="BERLIN",
    type=str,
)

parser.add_argument(
    "-i",
    "--input-dir",
    default="./traffic4cast2020/data/",
    type=str,
)

parser.add_argument(
    "-o",
    "--ouput-dir",
    default="./traffic4cast2020/",
    type=str,
)

parser.add_argument(
    "-t",
    "--test-slots",
    default="./traffic4cast2020/NeurIPS2020-traffic4cast/core-competition/util/test_slots.json",
    type=str,
)

parser.add_argument(
    "-m",
    "--max-size",
    default=int(5e10),
    type=int,
)
if __name__=="__main__":
    args = parser.parse_args()
    args.city = args.city.upper()
    file = glob.glob(os.path.join(args.input_dir,args.city,'*static*'))[0]
    f = h5py.File(file,'r')
    f = np.array(f['array'])
    f = np.moveaxis(f,-1,0)
    np.savez(f'./processed_data/{args.city}_static.npz',array=f)