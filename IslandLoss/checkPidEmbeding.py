
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

pubs_dict = data_utils.load_json(settings.GLOBAL_DATA_DIR, 'pubs_raw.json')
LMDB_NAME_EMB = "publication.emb.weighted"
lc_emb = LMDBClient(LMDB_NAME_EMB)
cnt = 0

for i, pid in enumerate(pubs_dict):
    if i % 1000 == 0:
        print ("idx: %d"%(i))
        emb = lc_emb.get(pid)
        if emb is None:
            print ("%s emb is null"%(pid))
            cnt =  cnt + 1

print ("cnt: %d"%(cnt))



