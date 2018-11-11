from __future__ import division, print_function
import tensorflow as tf
from tensorflow.python.training.adam import AdamOptimizer
import numpy as np

def ndmatmul(d3,d2):
    return tf.einsum("ijk,kl->ijl",d3,d2)

def softmax(acts, axis=None, reduction_indices=None):
    zero = tf.constant(0, dtype=tf.float32)
    where = tf.not_equal(acts, zero)
    exps = tf.where(where, tf.exp(acts), tf.zeros(tf.shape(acts)))
    return exps / tf.reduce_sum(exps, axis=axis, reduction_indices=reduction_indices, keepdims=True)


def softmax_padding(acts, padding, axis=None, reduction_indices=None):
    exps = tf.exp(acts-tf.stop_gradient(tf.reduce_max(acts,axis=axis,reduction_indices=reduction_indices,keepdims=True)))*padding
    return exps / tf.reduce_sum(exps, axis=axis, reduction_indices=reduction_indices, keepdims=True)


def perc_padding(acts, padding, axis=None, reduction_indices=None):
    exps = (acts - tf.reduce_min(acts, axis=axis, reduction_indices=reduction_indices, keepdims=True)) * padding
    return exps / tf.reduce_sum(exps, axis=axis, reduction_indices=reduction_indices, keepdims=True)


class CNN_Factory:
    def get_class(self, use_attention=False, att_max_value=0.7):
        class CNN_prior:
            def get_attention(self):
                self.pos_attention = tf.reduce_sum(tf.gradients(self.pre_max[:, 1], self.vectors)[0] * self.vectors, axis=2)
                self.pos_attention = softmax_padding(self.pos_attention, self.padding, axis=1)

                self.neg_attention = tf.reduce_sum(tf.gradients(self.pre_max[:, 0], self.vectors)[0] * self.vectors, axis=2)
                self.neg_attention = softmax_padding(self.neg_attention, self.padding, axis=1)

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

            def groupby(self,att):
                return ndmatmul(self.group_by,att)

            def __init__(self, word_vector_size):
                tf.reset_default_graph()
                self.vector_size = word_vector_size

                self.vectors = tf.placeholder(tf.float32, shape=(None, None, word_vector_size))
                self.user_terms = tf.placeholder(tf.float32, shape=(None, None))
                self.ut2 = tf.placeholder(tf.float32, shape=(None, None))
                self.group_by = tf.placeholder(tf.float32, shape=(None, None, None))
                self.padding = tf.placeholder(tf.float32, shape=(None, None))
                self.output = tf.placeholder(tf.float32, shape=(None, 1))
                self.dropout_rate = tf.placeholder(tf.float32)

                xavier = tf.contrib.layers.xavier_initializer()

                # 50 tri-gram, 50 4-gram and 50 5-gram
                filter_tri = tf.Variable(xavier((1, 2, word_vector_size, 50)), name="weight")  #
                bias_tri = tf.Variable(tf.zeros((1, 50)), name="bias")  #
                self.f3 = filter_tri
                self.b3 = bias_tri

                filter_4 = tf.Variable(xavier((1, 3, word_vector_size, 50)), name="weight")  #
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

                ut = tf.expand_dims(self.ut2, 2)  # NWC
                rel_masked, pre_max_true_masked_dropped, _ = self.forward(self.vectors * ut)
                self.rel_masked = rel_masked

                self.pre_max = pre_max_sum
                self.get_attention()

                # true_attention_error = 0.0
                att_reg = 0.0

                prediction_error = -tf.reduce_sum((self.output * tf.log(rel[:, 1] + 10 ** -5, name="log2rel") + (
                        1 - self.output) * tf.log(rel[:, 0] + 10 ** -5, name="log3rel")))

                # N, num_unique, text_length ; N,text_length
                pos_attention = tf.squeeze(tf.matmul(self.group_by, tf.expand_dims(self.pos_attention, -1)),
                                       squeeze_dims=-1)
                neg_attention = tf.squeeze(tf.matmul(self.group_by, tf.expand_dims(self.neg_attention, -1)),
                                       squeeze_dims=-1)
                self.pos_att_grouped = pos_attention
                self.neg_att_grouped = neg_attention

                pos_heads = tf.reduce_sum(tf.multiply(pos_attention, self.user_terms), axis=1)
                neg_heads = tf.reduce_sum(tf.multiply(neg_attention, self.user_terms), axis=1)
                self.pos_heads = pos_heads

                attention_error = 0.0
                occlusion_error = 0.0
                if use_attention:
                    attention_error += tf.reduce_sum(self.output*(pos_heads - 0.5) ** 2)
                    att_reg = tf.reduce_sum(self.output * tf.nn.relu(self.pos_attention - att_max_value)
                                                     + (1-self.output) * tf.nn.relu(self.neg_attention-att_max_value))
                    occlusion_error =  -tf.reduce_sum((self.output * tf.log(rel_masked[:, 1] + 10 ** -5, name="log2rel2") + (
                        1 - self.output) * tf.log(rel_masked[:, 0] + 10 ** -5, name="log3rel2")))


                self.att = attention_error

                self.error = (   prediction_error
                              + tf.sign(tf.reduce_sum(self.user_terms)) * attention_error
                              +  tf.sign(tf.reduce_sum(self.user_terms)) * occlusion_error
                              + tf.sign(tf.reduce_sum(self.user_terms)) * att_reg)

                self.a = tf.check_numerics(attention_error, message="att") + tf.check_numerics(pos_heads,
                                                                                               message="pos-heads") + tf.check_numerics(
                    neg_heads, message="neg-heads")
                self.opt = AdamOptimizer()
                self.optimizer = self.opt.minimize(self.error)
                self.uncertainty = 1

                self.sess = tf.Session()
                self.sess.run(tf.global_variables_initializer())
                self.n_trained = 0
                self.training = False

            def get_feed_dict(self, doc):
                return {self.vectors: np.array(doc.vectors, dtype=np.float32).reshape([1, -1, self.vector_size]),
                        self.output: [[doc.class_ * 1]],
                        self.user_terms: np.array(doc.user_terms, dtype=np.float32).reshape([1, -1]),
                        self.padding: np.array([1 for i in doc.words]).reshape([1, -1])}

            def blow_up(self,mat,num_rows,num_cols):
                blowed_mat = [i+[0]*(num_cols-len(i)) for i in mat]
                x=([0] * num_cols) * (num_rows - len(blowed_mat))
                if x:
                    blowed_mat.append(x)
                return blowed_mat

            def get_feed_dict_multiple(self, docs):
                dp = 0.7 if self.training else 1
                maximum = max([len(doc.vectors) for doc in docs])
                maximum = max([maximum,7])
                max_terms = max([len(doc.user_terms) for doc in docs])
                return {self.vectors: np.array(
                    [doc.vectors[:maximum] + [[0] * (self.vector_size)] * (maximum - len(doc.vectors[:maximum])) for doc
                     in
                     docs]).reshape([-1, maximum, self.vector_size]),
                        self.group_by:np.array([self.blow_up(doc.gb,max_terms,maximum) for doc in docs]),
                        self.ut2: np.array(
                            [doc.ut2[:maximum] + [0] * (maximum - len(doc.ut2[:maximum])) for doc in
                             docs]).reshape([-1, maximum]),
                        self.output: [[doc.class_ * 1] for doc in docs],
                        self.user_terms: np.array(
                            [doc.user_terms[:max_terms] + [0] * (max_terms - len(doc.user_terms[:max_terms])) for doc in
                             docs]).reshape([-1, max_terms]),
                        self.padding: np.array(
                            [[1] * len(doc.vectors[:maximum]) + [0] * (maximum - len(doc.vectors[:maximum])) for doc in
                             docs]).reshape([-1, maximum]),
                        self.dropout_rate:dp}

            def load(self, filename):
                saver = tf.train.Saver()
                saver.restore(self.sess, filename)
                pass

            def train(self, docs, train_full=False):
                self.training = True
                self.sess.run(tf.global_variables_initializer())
                sess = self.sess
                print("====23")
                n = len(docs)
                epochs = 200
                if train_full:
                    epochs = 10
                self.n_trained = n
                import random
                random.shuffle(docs)
                last_10 = [100] * 10
                prev_error = None
                for epoch in range(epochs):
                    total_error = 0
                    for doc_s in [docs[i:i + 1] for i in range(0, len(docs), 1)]:
                        fd = self.get_feed_dict_multiple(doc_s)
                        try:
                            sess.run(self.a, feed_dict=fd)
                        except Exception as e:
                            print("check")
                        _, error = sess.run([self.optimizer, self.error], feed_dict=fd)
                        # print(x,y)
                        # if epoch>50 and x>=0.5:
                        #     print("ch")
                        # print(error,error-x,x)
                        total_error += error
                    total_error = total_error / len(docs)
                    # print(total_error)
                    if train_full:
                        saver = tf.train.Saver()
                        saver.save(sess, "./{}.pkl".format(epoch))
                    # print(total_error)
                    if epoch>10 and total_error > 4:
                        self.train(docs)
                        return
                    last_10.pop(0)
                    last_10.append(total_error)
                    if max(last_10) < 0.05:
                        print("breaking")
                        break
                print(total_error)
                self.training = False

            def run(self, docs):
                sess = self.sess
                for doc_s in [docs[i:i + 1] for i in range(0, len(docs), 1)]:
                    fd = self.get_feed_dict_multiple(doc_s)

                    try:
                        l1 = sess.run([self.relevance, self.pos_att_grouped, self.neg_att_grouped,self.pos_heads],
                                  feed_dict=fd)
                    except Exception as e:
                        print("here")
                    for ind, doc in enumerate(doc_s):
                        d = {
                            "rel": l1[0][ind],
                            "pos_att": l1[1][ind],
                            "neg_att": l1[2][ind],
                            "pos_heads": l1[3][ind]
                        }
                        doc.pred_class = 0 if d["rel"] < 0.5 else 1
                        doc.parameters = d

        return CNN_prior