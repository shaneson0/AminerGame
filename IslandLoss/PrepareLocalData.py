

from utils.cache import LMDBClient
import tensorflow as tf
from os.path import join
from utils import settings
import numpy as np
from IslandLoss.prepareTrainData import prepareData
from utils import EmbedingCheck
from keras import backend as K

LMDB_NAME = "author_IslandLoss.emb.weighted"
lc = LMDBClient(LMDB_NAME)

TrainX, TrainY, TestX, TestY, NumberOfClass, AllX, Ally, pids = prepareData()
MeanAllX = np.mean(AllX, axis=0)

Embedding = 100
NUM_CLASSES = NumberOfClass
CENTER_LOSS_ALPHA = 0.0001
Island_Loss_ALPHA = 1.0
ratio = 0.0001
epochs = 3000

def l2Norm(x):
    return K.l2_normalize(x, axis=-1)

with tf.name_scope('input'):
    input_images = tf.placeholder(tf.float32, shape=(None,Embedding), name='input_images')
    labels = tf.placeholder(tf.int64, shape=(None), name='labels')

global_step = tf.Variable(0, trainable=False, name='global_step')

layer1 = tf.keras.layers.Dense(100, activation='relu', name='first_emb_layer',  kernel_regularizer=tf.keras.regularizers.l2(0.01))(input_images)
layer1 = tf.keras.layers.Dropout(0.6)(layer1)

layer2 = tf.keras.layers.Dense(64, activation='relu', name='last_emb_layer', kernel_regularizer=tf.keras.regularizers.l2(0.01))(layer1)
layer2 = tf.keras.layers.Dropout(0.6)(layer2)


feature = tf.keras.layers.Lambda(l2Norm, name='norm_layer', output_shape=[32])(layer2)
logits = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(feature)


export_path = join(settings.ISLAND_LOSS_DIR, "feature_model")


saver = tf.train.Saver()
with tf.Session() as sess:
    saver.save(sess, join(settings.ISLAND_LOSS_DIR, "Model", "feature_model_0.627907"))
    # saver.restore(sess, join(settings.ISLAND_LOSS_DIR, "feature_model"))

    Features = sess.run(feature, feed_dict={
        input_images: AllX - MeanAllX,
        labels: Ally
    })

    for idx, pid in enumerate(pids):
        emb = Features[idx]
        lc.set(pid, emb)


    # EmbedingCheck.check(Features, TestY, name="train2_embedding.jpg")


