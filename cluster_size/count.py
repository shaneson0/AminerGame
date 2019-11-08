from os.path import join
import numpy as np
import keras.backend as K
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Bidirectional
from utils.cache import LMDBClient
from utils import data_utils
from utils import settings

# LMDB_NAME = "author_100.emb.weighted"
LMDB_NAME = "author_triplets.emb"
lc = LMDBClient(LMDB_NAME)

data_cache = {}


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


def root_mean_log_squared_error(y_true, y_pred):
    first_log = K.log(K.clip(y_pred, K.epsilon(), np.inf) + 1.)
    second_log = K.log(K.clip(y_true, K.epsilon(), np.inf) + 1.)
    return K.sqrt(K.mean(K.square(first_log - second_log), axis=-1))


def create_model():
    model = Sequential()
    model.add(Bidirectional(LSTM(64), input_shape=(300, 100)))
    model.add(Dropout(0.5))
    model.add(Dense(1))

    model.compile(loss="msle",
                  optimizer='rmsprop',
                  metrics=[root_mean_squared_error, "accuracy", "msle", root_mean_log_squared_error])

    return model


def sampler(clusters, k=300, batch_size=10, min=1, max=300, flatten=False):
    xs, ys = [], []
    for b in range(batch_size):
        num_clusters = np.random.randint(min, max)
        sampled_clusters = np.random.choice(len(clusters), num_clusters, replace=False)
        items = []
        for c in sampled_clusters:
            items.extend(clusters[c])
        sampled_points = [items[p] for p in np.random.choice(len(items), k, replace=True)]
        x = []
        for p in sampled_points:
            if p in data_cache:
                x.append(data_cache[p])
            else:
                print("a")
                x.append(lc.get(p))
        if flatten:
            xs.append(np.sum(x, axis=0))
        else:
            xs.append(np.stack(x))
        ys.append(num_clusters)
    return np.stack(xs), np.stack(ys)


def gen_train(clusters, k=300, batch_size=1000, flatten=False):
    while True:
        yield sampler(clusters, k, batch_size, flatten=flatten)

def gen_sna(k=300):
    name_to_pubs_test = data_utils.load_json(settings.SNA_PUB_DIR, 'sna_valid_author_raw.json')
    xs = []
    names = []

    for name in name_to_pubs_test:
        names.append(name)
        x = []
        items = []
        for pid in name_to_pubs_test[name]:
            if lc.get(pid) is not None:
                items.append(pid)

        if len(items) == 0:
            continue

        sampled_points = [items[p] for p in np.random.choice(len(items), k, replace=True)]
        for p in sampled_points:
            emb = lc.get(p)
            # print ("emb: len :", len(emb))
            x.append(emb)
        xs.append(np.stack(x))
    xs = np.stack(xs)
    return names, xs

def gen_test(k=300, flatten=False):
    name_to_pubs_test = data_utils.load_json(settings.GLOBAL_DATA_DIR, 'name_to_pubs_test.json')
    xs, ys = [], []
    names = []
    print (name_to_pubs_test)
    for name in name_to_pubs_test:
        print ("name: ", name)
        names.append(name)
        num_clusters = len(name_to_pubs_test[name])
        x = []
        items = []
        for c in name_to_pubs_test[name]:  # one person
            for item in name_to_pubs_test[name][c]:
                if lc.get(item) is None:
                    continue
                items.append(item)
        if len(items) == 0:
            continue

        sampled_points = [items[p] for p in np.random.choice(len(items), k, replace=True)]
        for p in sampled_points:
            x.append(lc.get(p))
        print ("name: ", name,  "x: ", x)
        if flatten:
            xs.append(np.sum(x, axis=0))
        else:
            xs.append(np.stack(x))
        ys.append(num_clusters)
    xs = np.stack(xs)
    ys = np.stack(ys)
    return names, xs, ys


def run_rnn(k=300, seed=1106):
    name_to_pubs_train = data_utils.load_json(settings.GLOBAL_DATA_DIR, 'name_to_pubs_train.json')
    test_names, test_x, test_y = gen_test(k)
    # test_names, test_x = gen_sna(k)
    np.random.seed(seed)
    clusters = []
    for domain in name_to_pubs_train.values():
        for cluster in domain.values():
            clusters.append(cluster)

    for i, c in enumerate(clusters):
        if i % 100 == 0:
            print(i, len(c), len(clusters))
        for pid in c:
            data_cache[pid] = lc.get(pid)
    model = create_model()
    # print(model.summary())
    model.fit_generator(gen_train(clusters, k=300, batch_size=1000), steps_per_epoch=100, epochs=1000, validation_data=(test_x, test_y))
    kk = model.predict(test_x)
    wf = open(join(settings.OUT_DIR, 'n_clusters_rnn.txt'), 'w')
    for i, name in enumerate(test_names):
        wf.write('{}\t{}\t{}\n'.format(name, kk[i][0],test_y[i]))
    wf.close()


if __name__ == '__main__':
    run_rnn()
