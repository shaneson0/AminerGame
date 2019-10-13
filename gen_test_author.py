from os.path import join
import json
from utils import settings

with open(join(settings.SNA_PUB_DIR, "sna_valid_author_raw.json")) as fp:
    sna = json.load(fp)
    print (sna)
    fp.close()

res = {}
for name in sna.keys():
    res[name] = {name: sna[name]}


with open(join(settings.SNA_PUB_DIR, "name_to_pubs_test.json"), 'w') as fp:
    json.dump(res, fp)
    print (res)
    fp.close()

