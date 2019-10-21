from utils import data_utils
from utils import settings
import codecs
from os.path import join
from datetime import datetime
from utils import feature_utils
from global_.embedding import EmbeddingModel
from utils.cache import LMDBClient
from collections import defaultdict as dd
from sklearn.model_selection import train_test_split
import math
import numpy as np
import json


# prepare Data
TrainPids = []
with open(join(settings.TRAIN_PUB_DIR, "train_author.json"), "r") as fp:
    train_author = json.load(fp)
    fp.close()
for name in train_author.keys():
    for aid in train_author[name]:
        TrainPids = TrainPids + train_author[name][aid]

TrainPids = np.array(TrainPids)

TrainPids, TestPids = train_test_split(TrainPids, test_size=0.33, random_state=42)
PUBLICATION_LABEL = 'Publication.label'
lc_publication_label = LMDBClient(PUBLICATION_LABEL)

LMDB_NAME_EMB = "publication.emb.weighted"
lc_emb = LMDBClient(LMDB_NAME_EMB)

TrainX = []
TrainY = []


for pid in TrainPids:
    emb = lc_emb.get(pid)
    label = lc_emb.get(pid)
    print ("pid: ", pid, ", label: ", label)
    if emb is None:
        continue
    TrainX.append(emb)
    TrainY.append()

print (TrainPids, len(TrainPids))
print (TestPids, len(TestPids))














