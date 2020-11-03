from .config import CfgNode

_C = CfgNode()
_C.SEED = 1024

# dataset
_C.DATASET = CfgNode()
_C.DATASET.NAME = "BaseDataset"
_C.DATASET.OUTPUT_CHANNELS = 48
_C.DATASET.INPUT_CHANNELS = 108
_C.DATASET.DATA_PATH = "./processed_data/"
_C.DATASET.VALID_INDEX = ""
_C.DATASET.TEST_INDEX = ""
_C.DATASET.DAYS = [-7, -3, -2, -1, 1, 2, 3, 7]
_C.DATASET.HOLIDAYS = "./traffic4cast2020/processed_data/holiday.joblib"
_C.DATASET.USE_POS_WEIGHT = False
_C.DATASET.WEATHER = "./traffic4cast2020/processed_data/"
_C.DATASET.SUN = "./traffic4cast2020/processed_data/"

# _C.DATASET.EMBED_SCALE = 1
# model
_C.MODEL = CfgNode()
_C.MODEL.NAME = "hrnet"
_C.MODEL.MODEL_VERSION = "HighResolutionNet"
_C.MODEL.HIDDEN_ACTIVATION = "default"
_C.MODEL.USE_POS_WEIGHT = False
_C.MODEL.POS_WEIGHT = 33.0
_C.MODEL.MODEL_CONFIG_FILE = ""
_C.MODEL.GEO_NUM_EMBED = 215820
_C.MODEL.GEO_EMBED_DIM = 8
_C.MODEL.FROZEN_LAYERS = False

# torch
_C.TORCH = CfgNode()
_C.TORCH.NUM_THREADS = 4

# dist
_C.DIST = CfgNode()
_C.DIST.VERSION = "v1-hrnet"
_C.DIST.CHECKPOINT_PATH = "./traffic4cast2020/weights/"
_C.DIST.FORMATER = "{version}-{city}-{epoch}.pth"
_C.DIST.STORE_FREC = 1
_C.DIST.PRETRAIN_MODEL = "/bigdata/"

# log
_C.LOG = CfgNode()
_C.LOG.DIR = ""
_C.LOG.STEP = 10

# optimizer
_C.OPTIM = CfgNode()
_C.OPTIM.NAME = "FusedLAMB"
_C.OPTIM.INIT_LR = 1e-2
_C.OPTIM.USE_LR_SCHEDULER = True
_C.OPTIM.LR_SCHEDULER_TYPE = "get_linear_schedule_with_warmup"
_C.OPTIM.ADAM_EPSILON = 1e-8
_C.OPTIM.BATCH_SIZE = 8
_C.OPTIM.MAX_EPOCH = 40
_C.OPTIM.WARM_UP_EPOCH = 1.0
_C.OPTIM.EXPONENT = 0.999
_C.OPTIM.SGD_MOMENTUM = 0.9


def get_cfg(config=None, check_name=True):
    C = _C.clone()
    if config is not None:
        C.merge_from_file(config)
    config_name = C.DIST.VERSION
    config_file_name = config.split("/")[-1].split(".")[0]
    if check_name:
        assert (
            config_name == config_file_name
        ), "config name is different from version name"
    return C
