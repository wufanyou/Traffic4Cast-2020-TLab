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

#args = parser.parse_args(['-c','berlin'])


if __name__=='__main__':
    args = parser.parse_args()
    args.city = args.city.upper()
    files = glob.glob(os.path.join(args.input_dir,args.city,'*/*.h5'))
    #files = glob.glob(f'{args.input_dir}/{args.city}/*/*.h5')
    files = pd.DataFrame(files)
    files['dataset'] =  files[0].apply(lambda x:x.split('/')[-2])
    files['date'] = files[0].apply(lambda x:x.split('/')[-1].split('_')[0])
    files['date'] = pd.to_datetime(files.date)
    files = files.sort_values('date').reset_index(drop=True)
    with open(args.test_slots) as f:
        test_json = json.load(f)
    test_json = {list(d.keys())[0]:list(d.values())[0] for d in test_json}
    env = lmdb.open(os.path.join(args.ouput_dir,args.city), map_size=args.max_size)

    for day in tqdm(pd.date_range('2019-01-01','2019-12-31')):
        row = files[files['date']==day].reset_index(drop=True)
        if len(row)==0:
            f = np.zeros([288, 495, 436, 9]).astype(np.uint8)
        else:
            row = row.T[0]
            if row.dataset=='testing':
                f = np.zeros([288, 495, 436, 9]).astype(np.uint8)
                date = str(day.date())
                t = h5py.File(row[0],'r')
                t = list(t['array'])
                for i,idx in enumerate(test_json[date]):
                    f[idx:idx+12] = t[i]
            else:
                f = h5py.File(row[0],'r')
                f = list(f['array'])
                #f = np.stack(f)

        base_index = (day.dayofyear-1)*288
        with env.begin(write=True) as txn:
            for i in range(288):
                str_id = str(base_index+i)
                d = f[i]
                d = d.reshape(-1)
                s = io.BytesIO()
                np.savez(s,x=np.where(d!=0)[0],y=d[d!=0])
                txn.put(str_id.encode('ascii'),s.getvalue())