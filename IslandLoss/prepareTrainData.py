from utils import data_utils
from utils import settings
import codecs
from os.path import join
from datetime import datetime
from utils import feature_utils
from global_.embedding import EmbeddingModel
from utils.cache import LMDBClient
from collections import defaultdict as dd

import math



train_author = data_utils.load_data(settings.TRAIN_PUB_DIR, 'train_author.json')
for name in train_author.keys():
    for pid in train_author[name]:















