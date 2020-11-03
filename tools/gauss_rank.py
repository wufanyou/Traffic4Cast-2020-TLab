#from utils import *
import torch
from tqdm import tqdm
import numpy as np
import io
import lmdb
import os
from collections import defaultdict
import pandas as pd
from scipy.special import erfinv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "-c",
    "--city",
    default="BERLIN",
    type=str,
)

parser.add_argument(
    "-p",
    "--path",
    default="/home/omnisky/processed_data/",
    type=str,
)

args = parser.parse_args()

def get_item(idx):
    idx = str(idx).encode("ascii")
    try:
        with env.begin() as txn:data = txn.get(idx)
        data = np.load(io.BytesIO(data))
        x = np.zeros(495 * 436 * 9, dtype=np.uint8)
        x[data["x"]] = data["y"]
        x = x.reshape([495, 436, 9])
        x = np.moveaxis(x, -1, 0)
    except:
        x = np.zeros([9, 495, 436], dtype=np.uint8)
    return x

counter = defaultdict(lambda:defaultdict(lambda:0))
env = lmdb.open(os.path.join(args.path, args.city.upper()), readonly=True)

if __name__=="__main__":
    for idx in tqdm(range(52104)):
        array = get_item(idx)
        for i in range(9):
            t = array[i]
            for x ,y in zip(*np.unique(t,return_counts=True)):
                counter[i][x] +=y

    all_arr = []
    for i in range(9):
        df = pd.DataFrame(counter[i],index=[0]).T
        df = df.reset_index()
        df = df.sort_values('index')
        df.set_index('index',inplace=True)

        df = df[1:-1]

        df['cum'] = df.cumsum().shift().fillna(0)

        df['rank'] = erfinv(df['cum']/(df['cum'].max()+1e3))
        df['rank'] = df['rank']/df['rank'].max()
        df['rank'] = (df['rank']+1)/3
        if i!=8:
            arr = df.reset_index()['rank'].values
            arr = np.append(np.append(0,arr),1)
        else:
            arr = np.zeros(256)
            for idx,row in df.iterrows():
                arr[idx] = row['rank']
            arr[-1] = 1
        all_arr.append(arr)
    np.savez(f'{args.city.upper()}_value_map.npz',array=all_arr)