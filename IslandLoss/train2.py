# coding=utf8
import os
import numpy as np
import tensorflow as tf
from keras import backend as K
from IslandLoss.prepareTrainData import prepareData
from utils import EmbedingCheck
from IslandLoss import get_lc_center_loss

def get_center_loss(features, labels, alpha, num_classes):
    """获取center loss及center的更新op

    Arguments:
        features: Tensor,表征样本特征,一般使用某个fc层的输出,shape应该为[batch_size, feature_length].
        labels: Tensor,表征样本label,非one-hot编码,shape应为[batch_size].
        alpha: 0-1之间的数字,控制样本类别中心的学习率,细节参考原文.
        num_classes: 整数,表明总共有多少个类别,网络分类输出有多少个神经元这里就取多少.

    Return：
        loss: Tensor,可与softmax loss相加作为总的loss进行优化.
        centers: Tensor,存储样本中心值的Tensor，仅查看样本中心存储的具体数值时有用.
        centers_update_op: op,用于更新样本中心的op，在训练时需要同时运行该op，否则样本中心不会更新
    """
    # 获取特征的维数，例如256维
    len_features = features.get_shape()[1]
    # 建立一个Variable,shape为[num_classes, len_features]，用于存储整个网络的样本中心，
    # 设置trainable=False是因为样本中心不是由梯度进行更新的
    centers = tf.get_variable('centers', [num_classes, len_features], dtype=tf.float32,
                              initializer=tf.constant_initializer(0), trainable=False)
    # 将label展开为一维的，输入如果已经是一维的，则该动作其实无必要
    labels = tf.reshape(labels, [-1])

    # 根据样本label,获取mini-batch中每一个样本对应的中心值
    centers_batch = tf.gather(centers, labels)
    # 计算loss
    loss = tf.nn.l2_loss(features - centers_batch)

    # 当前mini-batch的特征值与它们对应的中心值之间的差
    diff = centers_batch - features

    # 获取mini-batch中同一类别样本出现的次数,了解原理请参考原文公式(4)
    unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
    appear_times = tf.gather(unique_count, unique_idx)
    appear_times = tf.reshape(appear_times, [-1, 1])

    diff = diff / tf.cast((1 + appear_times), tf.float32)
    diff = alpha * diff

    centers_update_op = tf.scatter_sub(centers, labels, diff)

    return loss, centers, centers_update_op

def l2Norm(x):
    return K.l2_normalize(x, axis=-1)

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

TrainX, TrainY, ValidX, ValidY, NumberOfClass, AllX, Ally, pids, TestX, TestY = prepareData()

# TrainX, TrainY
mean_train_x = np.mean(TrainX, axis=0)
mean_test_x = np.mean(ValidX, axis=0)

TrainX = list(chunks(TrainX, 2000))
TrainY = list(chunks(TrainY, 2000))


print ("pass")

Embedding = 100
NUM_CLASSES = NumberOfClass
CENTER_LOSS_ALPHA = 0.00002
Island_Loss_ALPHA = 0.00002
ratio = 0.1
epochs = 3000

with tf.name_scope('input'):
    input_images = tf.placeholder(tf.float32, shape=(None,Embedding), name='input_images')
    labels = tf.placeholder(tf.int64, shape=(None), name='labels')

global_step = tf.Variable(0, trainable=False, name='global_step')

layer1 = tf.keras.layers.Dense(100, activation='relu', name='first_emb_layer',  kernel_regularizer=tf.keras.regularizers.l2(0.01))(input_images)
layer1 = tf.keras.layers.Dropout(0.6)(layer1)

layer2 = tf.keras.layers.Dense(64, activation='relu', name='last_emb_layer', kernel_regularizer=tf.keras.regularizers.l2(0.01))(layer1)
layer2 = tf.keras.layers.Dropout(0.6)(layer2)

feature = tf.keras.layers.Lambda(l2Norm, name='norm_layer', output_shape=[64])(layer2)
logits = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(feature)

with tf.name_scope('loss'):
    with tf.name_scope('center_loss'):
        # center_loss, centers, centers_update_op = get_center_loss(feature, labels, CENTER_LOSS_ALPHA, NUM_CLASSES)
        center_loss, centers, centers_update_op, _, _ = get_lc_center_loss(feature, labels, CENTER_LOSS_ALPHA, Island_Loss_ALPHA, NUM_CLASSES)
    with tf.name_scope('softmax_loss'):
        softmax_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))
    with tf.name_scope('total_loss'):
        total_loss = softmax_loss + ratio * center_loss
        # total_loss = softmax_loss

with tf.name_scope('acc'):
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(logits, 1), labels), tf.float32))

with tf.name_scope('loss/'):
    tf.summary.scalar('CenterLoss', center_loss)
    tf.summary.scalar('SoftmaxLoss', softmax_loss)
    tf.summary.scalar('TotalLoss', total_loss)


optimizer = tf.train.AdamOptimizer(0.01)


with tf.control_dependencies([centers_update_op]):
    train_op = optimizer.minimize(total_loss, global_step=global_step)

summary_op = tf.summary.merge_all()
sess = tf.Session()
sess.run(tf.global_variables_initializer())



step = sess.run(global_step)
vali_acc = 0.0
softmaxloss = 0.0
totalloss = 0.0
centerloss = 0.0
while step <= epochs:


    for batchid, batchX in enumerate(TrainX):
        batchy = TrainY[batchid]
    # batchX = TrainX
    # batchy = TrainY
        _, summary_str, train_acc, softmaxloss, totalloss, centerloss = sess.run(
            [train_op, summary_op, accuracy, softmax_loss, total_loss, center_loss],
            feed_dict={
                input_images: batchX - mean_train_x,
                labels: batchy,
            })


    vali_acc = sess.run(
        accuracy,
        feed_dict={
            input_images: ValidX - mean_test_x,
            labels: ValidY
        })
    print ("softmaxloss: ", softmaxloss, ",totalloss: ", totalloss, ', center_loss: ', centerloss)
    print(("===== step: {}, train_acc:{:.4f}, vali_acc:{:.4f} ====".
          format(step, train_acc, vali_acc)))

    step += 1

Features = sess.run(feature,         feed_dict={
            input_images: ValidX - mean_test_x,
            labels: ValidY
        })
f1 = EmbedingCheck.check(Features, ValidY, name="train2_embedding.jpg")


# save model

from os.path import join
from utils import settings
saver = tf.train.Saver()
path = join(settings.ISLAND_LOSS_DIR, "200", "vali_acc_%s"%(vali_acc), "feature_model")
saver.save(sess, path)
print (path)


print ("End..")

#softmaxloss:  4.063885 ,totalloss:  31.816998 , center_loss:  277.53113