
from os.path import join
import json
from utils import settings

with open(join(settings.TRAIN_PUB_DIR, "train_author.json"), "r") as fp:
    train_author = json.load(fp)
    fp.close()

name_to_pubs_test = {}
name_to_pubs_train = {}

len = len(train_author)
for idx, key in enumerate(train_author.keys()):
    author = train_author[key]
    if idx < 0.1 * len:
        name_to_pubs_test[key] = author
    else:
        name_to_pubs_train[key] = author

with open(join(settings.GLOBAL_DATA_DIR, "name_to_pubs_test.json"), "w") as fp:
    json.dump(name_to_pubs_test, fp)
    fp.close()

with open(join(settings.GLOBAL_DATA_DIR, "name_to_pubs_train.json"), "w") as fp:
    json.dump(name_to_pubs_train, fp)
    fp.close()

