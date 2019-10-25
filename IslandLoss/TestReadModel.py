import tensorflow as tf
from os.path import join
from utils import settings
import numpy as np
from IslandLoss.prepareTrainData import prepareData
from utils import EmbedingCheck


export_path = join(settings.ISLAND_LOSS_DIR, "feature_model")

TrainX, TrainY, TestX, TestY, NumberOfClass, AllX, Ally = prepareData()
mean_test_x = np.mean(TestX, axis=0)

with tf.Session(graph=tf.Graph()) as sess:
    tf.saved_model.loader.load(sess, export_path)
    graph = tf.get_default_graph()
    print(graph.get_operations())

    Features = sess.run('norm_layer', feed_dict={
        'input_images': TestX - mean_test_x,
        'labels': TestY
    })
    EmbedingCheck.check(Features, TestY, name="train2_embedding.jpg")
