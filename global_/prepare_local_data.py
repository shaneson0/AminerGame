from os.path import join
import os
import numpy as np
from numpy.random import shuffle
from global_.global_model import GlobalTripletModel
from utils.eval_utils import get_hidden_output
from utils.cache import LMDBClient
from utils import data_utils
from utils import settings

IDF_THRESHOLD = 32  # small data
# IDF_THRESHOLD = 10


def dump_emb():
    """
    dump hidden embedding via trained global model for local model to use
    """
    LMDB_NAME = "author_100.emb.weighted"
    lc_input = LMDBClient(LMDB_NAME)
    INTER_LMDB_NAME = 'author_triplets.emb'
    lc_inter = LMDBClient(INTER_LMDB_NAME)
    global_model = GlobalTripletModel(data_scale=10000000)
    trained_global_model = global_model.load_triplets_model()
    name_to_pubs_train = data_utils.load_json(settings.GLOBAL_DATA_DIR, 'name_to_pubs_train.json')
    for name in name_to_pubs_train:
        if name == "roger_williams":
            continue
        # print('name', name)
        name_data = name_to_pubs_train[name]
        embs_input = []
        pids = []
        for i, aid in enumerate(name_data.keys()):
            if len(name_data[aid]) < 5:  # n_pubs of current author is too small
                continue
            for pid in name_data[aid]:
                cur_emb = lc_input.get(pid)
                if cur_emb is None:
                    # print ("pid emb is null: ", pid)
                    continue

                embs_input.append(cur_emb)
                pids.append(pid)
        embs_input = np.stack(embs_input)
        inter_embs = get_hidden_output(trained_global_model, embs_input)
        for i, pid_ in enumerate(pids):
            lc_inter.set(pid_, inter_embs[i])

    name_to_pubs_test = data_utils.load_json(settings.GLOBAL_DATA_DIR, 'name_to_pubs_test.json')
    for name in name_to_pubs_test:
        if name == "roger_williams" or name == "j_yu":
            continue
        print('name', name)
        name_data = name_to_pubs_test[name]
        embs_input = []
        pids = []
        for i, aid in enumerate(name_data.keys()):
            if len(name_data[aid]) < 5:  # n_pubs of current author is too small
                continue
            for pid in name_data[aid]:
                cur_emb = lc_input.get(pid)
                if cur_emb is None:
                    print ("pid emb is null: ", pid)
                    continue

                embs_input.append(cur_emb)
                pids.append(pid)
        embs_input = np.stack(embs_input)
        inter_embs = get_hidden_output(trained_global_model, embs_input)
        for i, pid_ in enumerate(pids):
            lc_inter.set(pid_, inter_embs[i])

    sna_valid_author_raw = data_utils.load_json(settings.SNA_PUB_DIR, 'sna_valid_author_raw.json')
    for name in sna_valid_author_raw.keys():
        if name == "j_yu":
            continue
        print ("name: ", name)
        checkPids = sna_valid_author_raw[name]
        embs_input = []
        pids = []
        for pid in checkPids:
            cur_emb = lc_input.get(pid)
            if cur_emb is None:
                continue
            embs_input.append(cur_emb)
            pids.append(pid)
        embs_input = np.stack(embs_input)
        inter_embs = get_hidden_output(trained_global_model, embs_input)
        for i, pid in enumerate(pids):
            lc_inter.set(pid, inter_embs[i])

def dump_test_emb():
    LMDB_NAME = "author_100.emb.weighted"
    lc_input = LMDBClient(LMDB_NAME)
    INTER_LMDB_NAME = 'author_triplets.emb'
    lc_inter = LMDBClient(INTER_LMDB_NAME)
    global_model = GlobalTripletModel(data_scale=1000000)
    trained_global_model = global_model.load_triplets_model()

    sna_valid_author_raw = data_utils.load_json(settings.SNA_PUB_DIR, 'sna_valid_author_raw.json')
    for name in sna_valid_author_raw.keys():
        if name == "j_yu":
            continue
        print ("name: ", name)
        checkPids = sna_valid_author_raw[name]
        embs_input = []
        pids = []
        for pid in checkPids:
            cur_emb = lc_input.get(pid)
            if cur_emb is None:
                continue
            embs_input.append(cur_emb)
            pids.append(pid)
        embs_input = np.stack(embs_input)
        inter_embs = get_hidden_output(trained_global_model, embs_input)
        for i, pid in enumerate(pids):
            lc_inter.set(pid, inter_embs[i])




def gen_local_data(idf_threshold=10):
    """
    generate local data (including paper features and paper network) for each associated name
    :param idf_threshold: threshold for determining whether there exists an edge between two papers (for this demo we set 29)
    """
    name_to_pubs_test = data_utils.load_json(settings.GLOBAL_DATA_DIR, 'name_to_pubs_test.json')

    idf = data_utils.load_data(settings.GLOBAL_DATA_DIR, 'feature_idf.pkl')
    INTER_LMDB_NAME = 'author_triplets.emb'
    lc_inter = LMDBClient(INTER_LMDB_NAME)
    LMDB_AUTHOR_FEATURE = "pub_authors.feature"
    lc_feature = LMDBClient(LMDB_AUTHOR_FEATURE)
    graph_dir = join(settings.DATA_DIR, 'local', 'graph-{}'.format(idf_threshold))
    os.makedirs(graph_dir, exist_ok=True)
    for i, name in enumerate(name_to_pubs_test):

        cur_person_dict = name_to_pubs_test[name]
        pids_set = set()
        pids = []
        pids2label = {}

        # generate content
        wf_content = open(join(graph_dir, '{}_pubs_content.txt'.format(name)), 'w')
        for i, aid in enumerate(cur_person_dict):
            items = cur_person_dict[aid]
            if len(items) < 5:
                continue
            for pid in items:
                pids2label[pid] = aid
                pids.append(pid)

        if name == "li_guo":
            print(i, name, pids)

        tempids = []
        shuffle(pids)
        for pid in pids:
            cur_pub_emb = lc_inter.get(pid)
            if cur_pub_emb is not None:
                tempids.append(pid)
                cur_pub_emb = list(map(str, cur_pub_emb))
                pids_set.add(pid)
                wf_content.write('{}\t'.format(pid))
                wf_content.write('\t'.join(cur_pub_emb))
                wf_content.write('\t{}\n'.format(pids2label[pid]))
        wf_content.close()

        if name == "li_guo":
            print(i, name, tempids)
            print ("len1 : %d, len2: %d"%(len(pids), len(tempids)))

        # generate network
        pids_filter = list(pids_set)
        n_pubs = len(pids_filter)
        print('n_pubs', n_pubs)
        wf_network = open(join(graph_dir, '{}_pubs_network.txt'.format(name)), 'w')
        for i in range(n_pubs-1):
            if i % 10 == 0:
                print(i)
            author_feature1 = set(lc_feature.get(pids_filter[i]))
            for j in range(i+1, n_pubs):
                author_feature2 = set(lc_feature.get(pids_filter[j]))
                common_features = author_feature1.intersection(author_feature2)
                idf_sum = 0
                for f in common_features:
                    idf_sum += idf.get(f, idf_threshold)
                    # print(f, idf.get(f, idf_threshold))
                if idf_sum >= idf_threshold:
                    wf_network.write('{}\t{}\n'.format(pids_filter[i], pids_filter[j]))
        wf_network.close()

def gen_sna_data(idf_threshold=10):
    """
    generate local data (including paper features and paper network) for each associated name
    :param idf_threshold: threshold for determining whether there exists an edge between two papers (for this demo we set 29)
    """
    # name_to_pubs_test = data_utils.load_json(settings.GLOBAL_DATA_DIR, 'name_to_pubs_test.json')
    name_to_pubs_test = data_utils.load_json(settings.SNA_PUB_DIR, 'sna_valid_author_raw.json')

    idf = data_utils.load_data(settings.GLOBAL_DATA_DIR, 'feature_idf.pkl')
    INTER_LMDB_NAME = 'author_triplets.emb'
    # INTER_LMDB_NAME = 'author_IslandLoss.emb.weighted'
    lc_inter = LMDBClient(INTER_LMDB_NAME)
    LMDB_AUTHOR_FEATURE = "pub_authors.feature"
    lc_feature = LMDBClient(LMDB_AUTHOR_FEATURE)
    graph_dir = join(settings.DATA_DIR, 'local', 'graph-{}'.format(idf_threshold))
    os.makedirs(graph_dir, exist_ok=True)
    for i, name in enumerate(name_to_pubs_test):

        cur_person_dict = name_to_pubs_test[name]
        pids_set = set()
        pids = []
        pids2label = {}

        # generate content
        wf_content = open(join(graph_dir, '{}_pubs_content.txt'.format(name)), 'w')
        for i, pid in enumerate(cur_person_dict):
            pids2label[pid] = 1
            pids.append(pid)

        tempids = []
        shuffle(pids)
        for pid in pids:
            cur_pub_emb = lc_inter.get(pid)
            if cur_pub_emb is not None:
                tempids.append(pid)
                cur_pub_emb = list(map(str, cur_pub_emb))
                pids_set.add(pid)
                wf_content.write('{}\t'.format(pid))
                wf_content.write('\t'.join(cur_pub_emb))
                wf_content.write('\t{}\n'.format(pids2label[pid]))
        wf_content.close()


        print(i, name, tempids)
        print ("len1 : %d, len2: %d"%(len(pids), len(tempids)))

        # generate network
        pids_filter = list(pids_set)
        n_pubs = len(pids_filter)
        print('n_pubs', n_pubs)
        wf_network = open(join(graph_dir, '{}_pubs_network.txt'.format(name)), 'w')
        for i in range(n_pubs-1):
            if i % 10 == 0:
                print(i)
            else:
                author_feature1 = lc_feature.get(pids_filter[i])
                if author_feature1 is None:
                    continue
                else:
                    author_feature1 = set(author_feature1)
                for j in range(i+1, n_pubs):
                    # print("pids_filter[j]: ", pids_filter[j])
                    # print ("lc_feature.get(pids_filter[j]): ", lc_feature.get(pids_filter[j]))
                    author_feature2 = lc_feature.get(pids_filter[j])
                    if author_feature2 is None:
                        continue
                    author_feature2 = set(author_feature2)
                    common_features = author_feature1.intersection(author_feature2)
                    idf_sum = 0
                    for f in common_features:
                        idf_sum += idf.get(f, idf_threshold)
                        # print(f, idf.get(f, idf_threshold))
                    if idf_sum >= idf_threshold:
                        wf_network.write('{}\t{}\n'.format(pids_filter[i], pids_filter[j]))
        wf_network.close()



if __name__ == '__main__':
    # dump_inter_emb()
    dump_emb()
    dump_test_emb()
    gen_local_data(idf_threshold=IDF_THRESHOLD)
    # gen_sna_data(idf_threshold=IDF_THRESHOLD)
    print('done')
