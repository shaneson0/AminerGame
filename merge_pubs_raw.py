from os.path import join
import json
from utils import settings


# with open(join(settings.TRAIN_PUB_DIR, "train_pub.json"), 'r') as fp:
#     train_pubs = json.load(fp)
#     fp.close()
#
# with open(join(settings.SNA_PUB_DIR, "sna_valid_pub.json"), 'r') as fp:
#     test_pubs = json.load(fp)
#     fp.close()
#
# z = {**train_pubs, **test_pubs}
# with open(join(settings.GLOBAL_DATA_DIR, "pubs_raw.json"), 'w') as fp:
#     json.dump(z,fp)
#     fp.close()
# print (z)

with open(join(settings.GLOBAL_DATA_DIR, "pubs_raw.json"), 'r') as fp:
    pub_raw = json.load(fp)
    fp.close()

print (pub_raw['JIfXICrQ'])


