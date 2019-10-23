#coding=utf-8

import tensorflow as tf
from keras.layers import Dense, Input, Lambda
from global_.triplet import l2Norm, euclidean_distance, triplet_loss, accuracy
from keras.models import Model, model_from_json
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from utils import settings
from os.path import join
EMB_DIM = 64

class CenterLossModel(object):
    def __init__(self, alpha, num_classes, EMB_DIM = 100):
        # self.labels = labels
        self.num_classes = num_classes
        self.alpha = alpha
        self.placeholder = {
            'input': tf.placeholder(tf.float32, shape=(None, EMB_DIM)),
            'labels': tf.placeholder(tf.int32, shape=(None))
        }


    def get_center_loss(self, features, labels, alpha, num_classes):
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


    def buildModel(self):
        emb_input = self.placeholder['input']
        layer1 = tf.keras.layers.Dense(100, activation='relu', name='first_emb_layer')
        layer2 = tf.keras.layers.Dense(64, activation='relu', name='last_emb_layer')
        norm_layer = tf.keras.layers.Lambda(l2Norm, name='norm_layer', output_shape=[64])
        encoded_emb = norm_layer(layer2(layer1(emb_input)))

        return encoded_emb

    def buildOptimizer(self, encoded_emb):
        OneHotLabel = tf.one_hot(self.placeholder['labels'], 64)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels= OneHotLabel,
            logits=encoded_emb
        ))
        # centerloss, centers, centers_update_op = self.get_center_loss(encoded_emb, self.placeholder['labels'], self.alpha, self.num_classes)
        loss = loss

        optimizer = tf.train.AdagradOptimizer(learning_rate=0.01)
        # with tf.control_dependencies([centers_update_op]):
        train_op = optimizer.minimize(loss)

        _acc, acc_op = tf.metrics.accuracy(OneHotLabel,encoded_emb,name="my_metric")

        return loss, train_op, _acc, acc_op

    def getAcc(self, encoded_emb):
        OneHotLabel = tf.one_hot(self.placeholder['labels'], 64)

        acc, acc_op = tf.metrics.accuracy(labels=tf.argmax(OneHotLabel, 1),
                                          predictions=tf.argmax(encoded_emb, 1))
        return  acc, acc_op


    def tSNEAnanlyse(self, emb, labels, trueLabels, savepath=False):
        plt.figure()
        # labels = np.array(labels) + 2
        X_new = TSNE(learning_rate=100).fit_transform(emb)
        plt.subplot(2, 1, 1)
        plt.scatter(X_new[:, 0], X_new[:, 1], c=labels, marker='o')

        plt.subplot(2, 1, 2)
        plt.scatter(X_new[:, 0], X_new[:, 1], c=trueLabels, marker='o')

        plt.show()

        if savepath:
            plt.savefig(savepath)
        plt.close()




    def tarin(self, batchX, batchy, testX, testy, epochs=3000):

        model = self.buildModel()
        loss, opt, acc, acc_op = self.buildOptimizer(model)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            # Train model
            for epoch in range(epochs):
                # for batchid, batchX in enumerate(X):
                # batchy = y[batchid]

                # Construct feed dictionary
                feed_dict = {self.placeholder['input']: batchX, self.placeholder['labels']: batchy}
                # Run single weight update
                outs = sess.run([loss, opt, acc_op], feed_dict=feed_dict)
                # Compute average loss
                lossScale = outs[0]
                accScale = outs[2]

                print("Epoch:", '%04d' % (epoch + 1), "loss=", "{:.5f}".format(lossScale), ',acc = {:.5f}'.format(accScale))

            testacc = sess.run([acc], feed_dict={self.placeholder['input']: testX, self.placeholder['labels']: testy})
            print ("test acc: ", testacc)

            # check embedding
            embedding = sess.run(model, feed_dict={self.placeholder['input']: testX[:-50], self.placeholder['labels']: testy[:-50]})
            self.tSNEAnanlyse(embedding, labels=testy, trueLabels=testy, savepath=join(settings.OUT_DIR, "test_embedding.jpg"))


if __name__ == '__main__':
    model = CenterLossModel([1,2,3], 0.1)













