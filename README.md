# Traffic4Cast-2020-TLab
The 2nd place solution to Neurips 2020 Traffic4Cast competition. Check traffic4cast_2020.pdf for more details about our solution. 

# How to run our solution.

1. run `tools/process_lmdb.py` with three city:

```
INPUT_DIR = './';
python tools/process_lmdb.py --city BERLIN -i $INPUT_DIR -o ./processed_data/ --test-slots ./processed_data/test_slots.json    -m 50000000000;
python tools/process_lmdb.py --city ISTANBUL -i $INPUT_DIR -o ./processed_data/ --test-slots ./processed_data/test_slots.json  -m 100000000000;
python tools/process_lmdb.py --city MOSCOW -i $INPUT_DIR -o ./processed_data/ --test-slots ./processed_data/test_slots.json    -m 200000000000;
```
Here `-m` option decides the max size for lmdb. This value varies for different cities.

2. run `tools/agg_feat_lmdb.py` with three cities.
```
for CITY in BERLIN ISTANBUL MOSCOW;
do
  python tools/agg_feat_lmdb.py --city $CITY;
done
```

3. Download pre-trained model [here](https://1drv.ms/u/s!AiK3JSLEIEcGxVutqMS0s01T7czA?e=TtInHa) in pre-trained  weights directory and 
put it into processed_data dictionary or set config file `PRETRAIN_MODEL: ''`
4. Install [apex](https://github.com/NVIDIA/apex) and other requirement libraries in requirements.txt

5. run `gen.py` to generate single model submission. Our best single model use `v4-hrnet-w48-geo-embed-include-valid.yaml`, and the online score is 1.1761e-3.
First download model wegihts [here](https://1drv.ms/u/s!AiK3JSLEIEcGxVutqMS0s01T7czA?e=TtInHa) in weights directory and put it into local weights directory and run:

```
for CITY in BERLIN ISTANBUL MOSCOW;
do
python gen.py --city $CITY \
              --path ./config/v4-hrnet-w48-geo-embed-include-valid.yaml \
              --tag best \
              --test-slots ./processed_data/test_slots.json;
done
```


6. or run `gen_ensemble.py` to run ensemble models with uint8 output. We provide a version with a score of around 1.1667e-3. Our best score needs float output, and some files in Moscow will exceed the 20M limit. So we need also generate two versions and replace ~4 files in Moscow to reach the best score. First, download model weights [here](https://1drv.ms/u/s!AiK3JSLEIEcGxVutqMS0s01T7czA?e=TtInHa) in weights directory and put it into local weights directory, and run:
```
for CITY in BERLIN ISTANBUL MOSCOW;
do
python gen_ensemble.py -c $CITY -w "4,3,1,3,1,2,1,4" \
--use-all "0,0,0,0,0,0,0,1" \
--tag "best,best,best,best,best,best,best,best" \
--config "./config/v4-hrnet-w48-geo-embed-include-valid.yaml,\
./config/v4-hrnet-w48-include-valid.yaml,\
./config/v4-hrnet-include-valid.yaml,\
./config/v6-hrnet-sun-fix-dst-include-valid.yaml,\
./config/v6-hrnet-3D.yaml,\
./config/v6-unet-include-valid.yaml,\
./config/v4-hrnet-w48-geo-add-0-day-embed-include-valid.yaml,\
./config/v4-hrnet-w48-include-valid-all.yaml" 
done
```
