from local.gae.input_data import load_local_data
from utils.cluster import clustering
from local.gae.preprocessing import preprocess_graph, construct_feed_dict, \
    sparse_to_tuple, normalize_vectors, gen_train_edges, cal_pos_weight
from utils import settings
from utils.data_utils import load_json
from utils.eval_utils import pairwise_precision_recall_f1, cal_f1


def load_test_names():
    ValidAuthor = load_json(settings.SNA_PUB_DIR, 'sna_valid_author_raw.json')
    return list(ValidAuthor.keys())

def load_train_names():
    ValidAuthor = load_json(settings.TRAIN_PUB_DIR, 'train_author.json')
    return list(ValidAuthor.keys())

def main():
    names = load_test_names()
    Prec = 0
    Recall = 0
    F1 = 0
    for name in names:
        adj, features, labels, Ids = load_train_names(name=name)
        emb_norm = normalize_vectors(features)
        clusters_pred = clustering(emb_norm, features)
        prec, rec, f1 = pairwise_precision_recall_f1(clusters_pred, labels)
        print('pairwise precision', '{:.5f}'.format(prec),
              'recall', '{:.5f}'.format(rec),
              'f1', '{:.5f}'.format(f1))
        Prec += prec
        Recall += rec
        F1 += f1
    Prec = Prec / (1.0 * len(names))
    Recall = Recall / (1.0 * len(names))
    F1 = F1 / (1.0 * len(names))

    print('All pairwise precision', '{:.5f}'.format(Prec),
          'recall', '{:.5f}'.format(Recall),
          'f1', '{:.5f}'.format(F1))
