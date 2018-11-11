from __future__ import division, print_function
import tensorflow as tf
from tensorflow.python.training.adam import AdamOptimizer
import numpy as np
import random


def softmax_padding(acts, padding, axis=None, reduction_indices=None):
    """
    Computes softmax over acts along axis (or reduction_indices) given the padding.
    Padding is a binary matrix of same dimensions as that of acts.
    """
    exps = tf.exp(acts - tf.stop_gradient(
        tf.reduce_max(acts, axis=axis, reduction_indices=reduction_indices, keepdims=True))) * padding
    return exps / tf.reduce_sum(exps, axis=axis, reduction_indices=reduction_indices, keepdims=True)


class CNN_Factory:
    def get_class(self, use_attribution=False, att_max_value=0.8):
        class CNN_prior:
            def get_attribution(self):
                self.pos_attribution = tf.reduce_sum(tf.gradients(self.pre_max_sum[:, 1], self.vectors)[0] * self.vectors, axis=2)
                self.pos_attribution = softmax_padding(self.pos_attribution, self.padding, axis=1)

                self.neg_attribution = tf.reduce_sum(tf.gradients(self.pre_max_sum[:, 0], self.vectors)[0] * self.vectors, axis=2)
                self.neg_attribution = softmax_padding(self.neg_attribution, self.padding, axis=1)

            def forward(self, v):

                vectors2d = tf.expand_dims(v, 1)  # None x 1 x 200 x 300 ... NHWC

                conv1 = tf.nn.conv2d(
                    input=vectors2d,
                    filter=self.f3,
                    strides=[1, 1, 1, 1],
                    padding="VALID"
                )  # None x 1 x words x 50
                A1 = tf.nn.leaky_relu(conv1 + self.b3)

                self.a1 = A1
                conv2 = tf.nn.conv2d(
                    input=vectors2d,
                    filter=self.f4,
                    strides=[1, 1, 1, 1],
                    padding="VALID"
                )  # None x 1 x words x 50

                A2 = tf.nn.leaky_relu(conv2 + self.b4)
                self.a2 = A2

                conv3 = tf.nn.conv2d(
                    input=vectors2d,
                    filter=self.f5,
                    strides=[1, 1, 1, 1],
                    padding="VALID"
                )  # None x 1 x words x 5

                A3 = tf.nn.leaky_relu(conv3 + self.b5)

                max_A1_train = tf.reshape(tf.squeeze(tf.reduce_max(A1, 2)), [-1, 50])  # None x 5
                max_A2_train = tf.reshape(tf.squeeze(tf.reduce_max(A2, 2)), [-1, 50])  # None x 5
                max_A3_train = tf.reshape(tf.squeeze(tf.reduce_max(A3, 2)), [-1, 50])  # None x 5

                concat = tf.concat([max_A1_train, max_A2_train, max_A3_train], axis=1)
                concat_drop = tf.nn.dropout(concat,keep_prob=self.dropout_rate)
                pre_max_true_drop = tf.matmul(concat_drop, self.relevance_weight) + self.relevance_bias
                rel = tf.nn.softmax(pre_max_true_drop, axis=1)

                sum_A1_train = tf.reshape(tf.squeeze(tf.reduce_sum(A1, 2)), [-1, 50])  # None x 5
                sum_A2_train = tf.reshape(tf.squeeze(tf.reduce_sum(A2, 2)), [-1, 50])  # None x 5
                sum_A3_train = tf.reshape(tf.squeeze(tf.reduce_sum(A3, 2)), [-1, 50])  # None x 5

                concat_sums = tf.concat([sum_A1_train, sum_A2_train, sum_A3_train], axis=1)
                pre_max_sum = tf.matmul(concat_sums, self.relevance_weight) + self.relevance_bias
                return rel, pre_max_true_drop, pre_max_sum

            def __init__(self, word_vector_size):
                tf.reset_default_graph()
                self.vector_size = word_vector_size

                self.vectors = tf.placeholder(tf.float32, shape=(None, None, word_vector_size))
                self.user_terms = tf.placeholder(tf.float32, shape=(None, None))
                self.padding = tf.placeholder(tf.float32, shape=(None, None))
                self.output = tf.placeholder(tf.float32, shape=(None, 1))
                self.dropout_rate = tf.placeholder(tf.float32)

                xavier = tf.contrib.layers.xavier_initializer()

                # 50 tri-gram, 50 4-gram and 50 5-gram
                filter_tri = tf.Variable(xavier((1, 3, word_vector_size, 50)), name="weight")  #
                bias_tri = tf.Variable(tf.zeros((1, 50)), name="bias")  #
                self.f3 = filter_tri
                self.b3 = bias_tri

                filter_4 = tf.Variable(xavier((1, 4, word_vector_size, 50)), name="weight")  #
                bias_4 = tf.Variable(tf.zeros((1, 50)), name="bias")
                self.f4 = filter_4
                self.b4 = bias_4

                filter_5 = tf.Variable(xavier((1, 5, word_vector_size, 50)), name="weight")  #
                bias_5 = tf.Variable(tf.zeros((1, 50)), name="bias")
                self.f5 = filter_5
                self.b5 = bias_5

                with tf.name_scope("relevance"):
                    hidden = 150
                    self.relevance_weight = tf.Variable(0.01 * xavier((hidden, 2)))
                    self.relevance_bias = tf.Variable(0.0 * xavier((1, 2)))
                    self.relevance_attention_weight = tf.Variable(0.01 * xavier((100, 2)))
                    self.relevance_attention_bias = tf.Variable(0.0 * xavier((1, 2)))

                rel, pre_max_true_dropped, pre_max_sum = self.forward(self.vectors)
                self.relevance = rel[:, 1]

                ut = tf.expand_dims(self.user_terms, 2)  # NWC
                rel_masked, pre_max_true_masked_dropped, _ = self.forward(self.vectors * ut)
                self.rel_masked = rel_masked

                self.pre_max_sum = pre_max_sum
                self.get_attribution()

                prediction_error = -tf.reduce_sum((self.output * tf.log(rel[:, 1] + 10 ** -5, name="log2rel") + (
                        1 - self.output) * tf.log(rel[:, 0] + 10 ** -5, name="log3rel")))

                pos_heads = tf.reduce_sum(tf.multiply(self.pos_attribution, self.user_terms), axis=1)
                neg_heads = tf.reduce_sum(tf.multiply(self.neg_attribution, self.user_terms), axis=1)

                misattribution_error = 0.0
                corrective_error = 0.0

                if use_attribution:
                    misattribution_error += -tf.reduce_sum(self.output * pos_heads
                                                     + (1 - self.output) * neg_heads)

                    corrective_error = -tf.reduce_sum(
                        (self.output * tf.log(rel_masked[:, 1] + 10 ** -5, name="log2rel2") + (
                                1 - self.output) * tf.log(rel_masked[:, 0] + 10 ** -5, name="log3rel2")))


                self.error = (prediction_error
                              + tf.sign(tf.reduce_sum(self.user_terms))*(misattribution_error
                              + corrective_error))

                self.opt = AdamOptimizer()
                self.optimizer = self.opt.minimize(self.error)

                self.sess = tf.Session()
                self.sess.run(tf.global_variables_initializer())
                self.training = False

            def get_feed_dict_multiple(self, docs):
                dp = 0.7 if self.training else 1
                maximum = max([len(doc.vectors) for doc in docs]+[5])
                return {self.vectors: np.array(
                    [doc.vectors[:maximum] + [[0] * (self.vector_size)] * (maximum - len(doc.vectors[:maximum])) for doc
                     in
                     docs]).reshape([-1, maximum, self.vector_size]),
                        self.output: [[doc.class_ * 1] for doc in docs],
                        self.user_terms: np.array(
                            [doc.user_terms[:maximum] + [0] * (maximum - len(doc.user_terms[:maximum])) for doc in
                             docs]).reshape([-1, maximum]),
                        self.padding: np.array(
                            [[1] * len(doc.vectors[:maximum]) + [0] * (maximum - len(doc.vectors[:maximum])) for doc in
                             docs]).reshape([-1, maximum]),
                        self.dropout_rate: dp}

            def train(self, docs):
                self.training = True

                # Re-initialize the machine during every training round
                self.sess.run(tf.global_variables_initializer())
                sess = self.sess
                print("====")

                epochs = 200 # maximum training epochs
                random.shuffle(docs)

                last_10 = [100] * 10

                for epoch in range(epochs):
                    total_error = 0
                    # Stochastic Gradient Descent (mini-batch size = 1) works best.
                    for doc_s in [docs[i:i + 1] for i in range(0, len(docs), 1)]:
                        fd = self.get_feed_dict_multiple(doc_s)
                        # error=sess.run(self.error,feed_dict=fd)

                        _, error = sess.run([self.optimizer, self.error], feed_dict=fd)
                        total_error += error
                    total_error = total_error / len(docs)

                    if epoch > 10 and total_error > 4:
                        self.train(docs)
                        return
                    last_10.pop(0)
                    last_10.append(total_error)
                    if abs(last_10[-1]-last_10[-2]) < 10**-4:
                        print("breaking")
                        break
                print(total_error)
                self.training = False

            def run(self, docs):
                random.shuffle(docs)
                sess = self.sess
                num_correct = 0
                num_seen = 0
                for doc_s in [docs[i:i + 1] for i in range(0, len(docs), 1)]:
                    fd = self.get_feed_dict_multiple(doc_s)
                    l1 = sess.run([self.relevance, self.pos_attribution, self.neg_attribution], feed_dict=fd)
                    for ind, doc in enumerate(doc_s):
                        d = {
                            "rel": l1[0][ind],
                            "pos_att": l1[1][ind],
                            "neg_att": l1[2][ind]
                        }
                        doc.pred_class = 0 if d["rel"] < 0.5 else 1
                        doc.parameters = d
                        num_correct += 1*(doc.pred_class==doc.class_)
                        num_seen += 1
                    if num_seen%1000==0:
                        print(num_correct/num_seen*100)

        return CNN_prior
