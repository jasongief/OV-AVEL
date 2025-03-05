from easydict import EasyDict as edict
import yaml
import pdb

"""
default config
"""
cfg = edict()
cfg.TRAIN_BG_CLASS_ID = 46 # train
cfg.TOTAL_BG_CLASS_ID = 67 # total
cfg.DATA_BASE_DIR = "/root/autodl-tmp/OV_AVEL/ovave_dataset_preprocessed"
cfg.META_CSV_PATH = "/root/autodl-tmp/OV_AVEL/ovave_dataset_meta.csv"
# cfg.META_CSV_PATH = "/root/autodl-tmp/OV_AVEL/ovave_trainratio0.75_meta.csv"
cfg.ANNO_JSON_PATH = "/root/autodl-tmp/OV_AVEL/released_ovavel_dataset_anno.json"
cfg.TOTAL_CLOSE_OPEN_CATEGORY_CSV_PATH = "/root/autodl-tmp/OV_AVEL/ovave_total_close_open_categories.csv"
cfg.TRAIN_CLOSE_CATEGORY_CSV_PATH = "/root/autodl-tmp/OV_AVEL/ovave_train_close_categories.csv"

#### spatial attention
cfg.SA = edict()
cfg.SA.SA_ATTN_FLAG = False #!
cfg.SA.SA_ATTN_TYPE = "bothLast" # select from [bothFirst, bothLast, evenFirst, evenLast, fixedBlkids]
cfg.SA.SA_BOTH_LAYER_K = 1
cfg.SA.SA_EVEN_LAYER_K = 1
cfg.SA.SA_FIXED_LAYER_A = [0, 3]
cfg.SA.SA_FIXED_LAYER_V = [8, 16]

#### temporal attention
cfg.TA = edict()
cfg.TA.TA_ATTN_FLAG = True #!
cfg.TA.SA_LAYER_NUM = 1
cfg.TA.XA_LAYER_NUM = 1
cfg.TA.HIDDEN_DIM = 256
cfg.TA.FF_DIM = 512
cfg.TA.HEAD_NUM = 1
cfg.TA.DROPOUT = 0.1
cfg.TA.USE_ADJ_IN_ATTN = False
cfg.TA.GAMMA = 0.6
cfg.TA.BIAS = 0.2
cfg.TA.USE_MASK_IN_ATTN = False
cfg.TA.WIN_SIZE = 5
cfg.TA.NORM_FLAG = None

#### text projection tuning
cfg.TEXT_TUNE_FLAG = False