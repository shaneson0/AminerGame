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

TrainPids = []
train_author = data_utils.load_data(settings.TRAIN_PUB_DIR, 'train_author.json')
for name in train_author.keys():
    for aid in train_author[name]:
        for pids in train_author[name][aid]:
            TrainPids = TrainPids + pids

TrainPids = np.array(TrainPids)

TrainPids, TestPids = train_test_split(TrainPids, test_size=0.33, random_state=42)
print (TrainPids, len(TrainPids))
print (TestPids, len(TrainPids))














