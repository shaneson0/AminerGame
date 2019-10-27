from sklearn.manifold import TSNE
from utils import settings
from os.path import join
from utils.eval_utils import pairwise_precision_recall_f1, cal_f1
from utils.cluster import clustering
from local.gae.preprocessing import preprocess_graph, construct_feed_dict, \
    sparse_to_tuple, normalize_vectors, gen_train_edges, cal_pos_weight
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def tSNEAnanlyse(emb, labels, trueLabels, savepath=False):
    plt.figure()
    # labels = np.array(labels) + 2
    X_new = TSNE(learning_rate=100).fit_transform(emb)
    plt.subplot(1, 1, 1)
    plt.scatter(X_new[:, 0], X_new[:, 1], c=labels, marker='o')
    Points = list(zip(X_new[:, 0], X_new[:, 1]))
    for idx, point in enumerate(Points):
        x = point[0]
        y = point[1]
        label = labels[idx]
        plt.text(x, y+0.3, str(label), ha='center', va='bottom', fontsize=10.5)


    plt.show()

    if savepath:
        plt.savefig(savepath)
    plt.close()

from quick_cluster import FINCH
from sklearn.metrics import silhouette_score
def Testcheck(embedding, embeddingLabels, name):

    Maxscore = -10000
    NumberOfCluster = 0
    emb_norm = normalize_vectors(embedding)

    c, num_clust, req_c = FINCH(emb_norm)
    print("num_clust: ", num_clust)
    for nc in num_clust:
        emb_norm = normalize_vectors(emb_norm)
        TempLabels = clustering(emb_norm, nc)
        score = silhouette_score(emb_norm, TempLabels)
        print('nc: ', nc, ', score: ', score)
        if score > Maxscore:
            Maxscore = score
            # tClusterLabels = TempLabels
            NumberOfCluster = nc


    clusters_pred = list(clustering(emb_norm, num_clusters=NumberOfCluster))

    print ("clusters_pred: ", clusters_pred)
    print ("embeddingLabels: ", embeddingLabels)
    prec, rec, f1 = pairwise_precision_recall_f1(clusters_pred, embeddingLabels)
    print('pairwise precision', '{:.5f}'.format(prec),
          'recall', '{:.5f}'.format(rec),
          'f1', '{:.5f}'.format(f1))
    tSNEAnanlyse(embedding, labels=embeddingLabels, trueLabels=embeddingLabels, savepath=join(settings.OUT_DIR, name))



def check(embedding, embeddingLabels, name):
    emb_norm = normalize_vectors(embedding)
    clusters_pred = list(clustering(emb_norm, num_clusters=len(list(set(embeddingLabels)))))

    print ("clusters_pred: ", clusters_pred)
    print ("embeddingLabels: ", embeddingLabels)
    prec, rec, f1 = pairwise_precision_recall_f1(clusters_pred, embeddingLabels)
    print('pairwise precision', '{:.5f}'.format(prec),
          'recall', '{:.5f}'.format(rec),
          'f1', '{:.5f}'.format(f1))
    tSNEAnanlyse(embedding, labels=embeddingLabels, trueLabels=embeddingLabels, savepath=join(settings.OUT_DIR, name))
    return f1
