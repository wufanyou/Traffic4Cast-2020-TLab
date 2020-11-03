import lmdb
import os
import io
import numpy as np
from utils import *
from tqdm import tqdm
import torch
import pandas as pd
import json
import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    "-c",
    "--city",
    default="BERLIN",
    type=str,
)

args = parser.parse_args()
city = args.city.upper()
#cfg = get_cfg()
#cfg.merge_from_file('v5-hrnet-base.yaml')

test_slots = "./traffic4cast2020/NeurIPS2020-traffic4cast/core-competition/util/test_slots.json"
with open(test_slots) as f:
    test_json = json.load(f)
test_json = {list(d.keys())[0]:list(d.values())[0] for d in test_json}
test_json = {pd.to_datetime(k).dayofyear:v for k,v in test_json.items()}

val_dates = "./traffic4cast2020/NeurIPS2020-traffic4cast/core-competition/util/val_dates.json"
with open(val_dates) as f:
    valid_json = json.load(f)
valid_json = [pd.to_datetime(k).dayofyear for k in valid_json]

cipt_map = {'BERLIN':int(5e10),'ISTANBUL':int(1e11),'MOSCOW':int(2e11)}
env = lmdb.open(os.path.join('./traffic4cast2020/processed_data/', f"{city.upper()}_GK"),map_size = cipt_map[city])


def process(dayofyear):
    all_data = []   
    for idx in range((dayofyear-1)*288,dayofyear*288):
        idx = str(idx).encode("ascii")
        with env.begin() as txn:
            data = txn.get(idx)
        data = np.load(io.BytesIO(data))
        #print(data['y'])
        x = np.zeros(495 * 436 * 9, dtype=np.float32)
        x[data["x"]] = data["y"]
        x = x.reshape([495, 436, 9])
        x = np.moveaxis(x, -1, 0)
        all_data.append(x.copy())
    all_data = np.stack(all_data)

    if dayofyear<=181:
        for rand_idx in range(50):
            idx_num = np.random.choice(range(1, 6), p=[0.02, 0.23, 0.5, 0.23, 0.02])
            idx = np.random.randint(0, 60, [5]) + np.arange(0, 288, 60)
            idx = np.clip(idx, 0, 276)
            idx = np.random.choice(idx, idx_num)
            idx.sort()
            idx = np.array([i+j for i in idx for j in range(12)])
            value = torch.round(torch.tensor(all_data[idx]).float().mean(0)).cpu().numpy()
            d = value.reshape(-1)
            str_id = f"dayofyear/{dayofyear}/{rand_idx}"
            s = io.BytesIO()
            np.savez(s,x=np.where(d!=0)[0],y=d[d!=0])
            with env.begin(write=True) as txn:
                txn.put(str_id.encode('ascii'),s.getvalue())
                
    elif dayofyear in test_json:
        idx = np.array(test_json[dayofyear])
        idx = np.array([i+j for i in idx for j in range(12)])
        idx.sort()
        value = torch.round(torch.tensor(all_data[idx]).float().mean(0)).cpu().numpy()
        d = value.reshape(-1)
        str_id = f"dayofyear/{dayofyear}/0"
        s = io.BytesIO()
        np.savez(s,x=np.where(d!=0)[0],y=d[d!=0])
        with env.begin(write=True) as txn:
            txn.put(str_id.encode('ascii'),s.getvalue())
            
    elif dayofyear in valid_json:
        idx_num = np.random.choice(range(1, 6), p=[0.02, 0.23, 0.5, 0.23, 0.02])
        idx = np.random.randint(0, 60, [5]) + np.arange(0, 288, 60)
        idx = np.clip(idx, 0, 276)
        idx.sort()
        idx = np.array([i+j for i in idx for j in range(12)])
        value = torch.round(torch.tensor(all_data[idx]).float().mean(0)).cpu().numpy()
        d = value.reshape(-1)
        str_id = f"dayofyear/{dayofyear}/0"
        s = io.BytesIO()
        np.savez(s,x=np.where(d!=0)[0],y=d[d!=0])
        with env.begin(write=True) as txn:
            txn.put(str_id.encode('ascii'),s.getvalue())
    else:
        candidate_dayofyear = {279:276,278:276,277:280}[dayofyear] # 276 280 in test
        idx = np.array(test_json[candidate_dayofyear])
        idx = np.array([i+j for i in idx for j in range(12)])
        idx.sort()
        value = torch.clamp(torch.round(torch.tensor(all_data[idx]).float().mean(0)),0,255).cpu().numpy()
        d = value.reshape(-1)
        str_id = f"dayofyear/{dayofyear}/0"
        s = io.BytesIO()
        np.savez(s,x=np.where(d!=0)[0],y=d[d!=0])
        with env.begin(write=True) as txn:
            txn.put(str_id.encode('ascii'),s.getvalue())
        
if __name__=='__main__':
    for i in tqdm(range(1,182)):
        process(i)
    for i in tqdm(range(182,366)):
        process(i)
    #for i in [277,278,279]:
        #process(i)