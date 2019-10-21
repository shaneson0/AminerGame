from utils import data_utils
from utils import settings
import codecs
from os.path import join
import datetime
from utils import feature_utils

start_time = datetime.now()

def dump_paper_feature_to_file():
    pubs_dict = data_utils.load_json(settings.GLOBAL_DATA_DIR, 'pubs_raw.json')
    wf = codecs.open(join(settings.ISLAND_LOSS_DIR, 'paper_features.txt'), 'w', encoding='utf-8')
    for i, pid in enumerate(pubs_dict):
        if i % 1000 == 0:
            print(i, datetime.now()-start_time)
        paper = pubs_dict[pid]
        if "title" not in paper or "authors" not in paper:
            continue
        if len(paper["authors"]) > 30:
            print(i, pid, len(paper["authors"]))
        if len(paper["authors"]) > 100:
            continue
        author_feature = feature_utils.extract_author_features(paper, 0)
        pid = '{}'.format(pid)
        wf.write(pid + '\t' + ' '.join(author_feature) + '\n')
    wf.close()


if __name__ == '__main__':
    dump_paper_feature_to_file()

