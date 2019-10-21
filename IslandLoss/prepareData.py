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





def dump_author_features_to_cache():
    """
    dump author features to lmdb
    """
    LMDB_NAME = 'publication_IslandLoss.feature'
    lc = LMDBClient(LMDB_NAME)
    with codecs.open(join(settings.ISLAND_LOSS_DIR, 'paper_features.txt'), 'r', encoding='utf-8') as rf:
        for i, line in enumerate(rf):
            if i % 1000 == 0:
                print('line', i)
            items = line.rstrip().split('\t')
            pid_order = items[0]
            # print ("pid_order: ", pid_order, items)
            author_features = items[1].split()
            lc.set(pid_order, author_features)


def cal_feature_idf():
    """
    calculate word IDF (Inverse document frequency) using publication data
    """
    feature_dir = join(settings.DATA_DIR, 'IslandLoss')
    counter = dd(int)
    cnt = 0
    LMDB_NAME = 'publication_IslandLoss.feature'
    lc = LMDBClient(LMDB_NAME)
    author_cnt = 0
    with lc.db.begin() as txn:
        for k in txn.cursor():
            features = data_utils.deserialize_embedding(k[1])
            if author_cnt % 10000 == 0:
                print(author_cnt, features[0], counter.get(features[0]))
            author_cnt += 1
            for f in features:
                cnt += 1
                counter[f] += 1
    idf = {}
    for k in counter:
        idf[k] = math.log(cnt / counter[k])
    data_utils.dump_data(dict(idf), feature_dir, "IslandLoss_feature_idf.pkl")


def dump_author_embs():
    """
    dump author embedding to lmdb
    author embedding is calculated by weighted-average of word vectors with IDF
    """
    emb_model = EmbeddingModel.Instance()
    idf = data_utils.load_data(settings.ISLAND_LOSS_DIR, 'IslandLoss_feature_idf.pkl')
    print('idf loaded')
    LMDB_NAME_FEATURE = 'publication_IslandLoss.feature'
    lc_feature = LMDBClient(LMDB_NAME_FEATURE)
    LMDB_NAME_EMB = "publication.emb.weighted"
    lc_emb = LMDBClient(LMDB_NAME_EMB)
    cnt = 0
    with lc_feature.db.begin() as txn:
        for k in txn.cursor():
            if cnt % 1000 == 0:
                print('cnt', cnt, datetime.now()-start_time)
            cnt += 1
            pid = k[0].decode('utf-8')
            # print ("pid_order: ", pid_order)
            features = data_utils.deserialize_embedding(k[1])
            cur_emb = emb_model.project_embedding(features, idf)
            if cur_emb is not None:
                # print ("pid_order: is not none", pid_order)
                lc_emb.set(pid, cur_emb)

def encode_labels(train_author):
    labels = []
    for name in train_author.keys():
        for aid in train_author[name].keys():
            labels.append(aid)
    classes = set(labels)
    classes_dict = {c: i for i, c in enumerate(classes)}
    return classes_dict
    # return list(map(lambda x: classes_dict[x], labels))

def dumpPublicationLabel():
    PUBLICATION_LABEL = 'Publication.label'
    lc_publication_label = LMDBClient(PUBLICATION_LABEL)
    train_author = data_utils.load_data(settings.TRAIN_PUB_DIR, 'train_author.json')
    classes_dict = encode_labels(train_author)

    for name in train_author.keys():
        for aid in train_author[name].keys():
            for pid in train_author[name][aid]:
                print ("%s : %s"%(pid, classes_dict[aid]))
                lc_publication_label.set(pid, classes_dict[aid])

if __name__ == '__main__':
    # dump_paper_feature_to_file()
    # dump_author_features_to_cache()
    emb_model = EmbeddingModel.Instance()
    emb_model.train('shanxuan_islandLoss', LMDB_NAME='publication_IslandLoss.feature')  # training word embedding model
    cal_feature_idf()
    dump_author_embs()
    dumpPublicationLabel()



