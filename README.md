# Traffic4Cast-2020-TLab
The 2nd place solution to Neurips 2020 Traffic4Cast competition. Check traffic4cast_2020.pdf for more details about our solution. The usage of this repository will be updated later. 

# Instruction of how to run the code.

1. run `tools/process_lmdb.py` with three city.
2. run `tools/agg_feat_lmdb.py` with three city.
3. Download pretrained model [here](https://1drv.ms/u/s!AiK3JSLEIEcGxVutqMS0s01T7czA?e=bJpWDP) in pretrained weights directory and 
put it into processed_data dictionary or set config file `PRETRAIN_MODEL: ''`
4. Install [apex](https://github.com/NVIDIA/apex) and other requirement libraries in requirements.txt

5. run `gen.py` to geneante single model submision, our best single model use `v4-hrnet-w48-geo-embed-include-valid.yaml` and the online score is 1.1761e-3.
First download model wegihts [here](https://1drv.ms/u/s!AiK3JSLEIEcGxVutqMS0s01T7czA?e=bJpWDP) in weights directory and put it into weights directory and run:

```
for CITY in BERLIN ISTANBUL MOSCOW;
do
python gen.py --city $CITY --path ./config/v4-hrnet-w48-geo-embed-include-valid.yaml --tag best --test-slots ./processed_data/test_slots.json;
done
```


6. or run `gen-ensemble.py` to run ensemble models.
