from os.path import abspath, dirname, join
import os

PROJ_DIR = join(abspath(dirname(__file__)), '..')
DATA_DIR = join(PROJ_DIR, 'data')
OUT_DIR = join(PROJ_DIR, 'out')
EMB_DATA_DIR = join(DATA_DIR, 'emb')
GLOBAL_DATA_DIR = join(DATA_DIR, 'global_')
ISLAND_LOSS_DIR = join(DATA_DIR, 'IslandLoss')
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(EMB_DATA_DIR, exist_ok=True)


TRAIN_PUB_DIR = join(DATA_DIR, 'train')
SNA_PUB_DIR = join(DATA_DIR, 'sna')

