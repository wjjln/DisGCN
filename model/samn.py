import tensorflow as tf
import numpy as np
import os
from base_model import base_model

class samn(base_model):
    def inputSupply(self, data_dict):
        self.social_edges_user0 = data_dict['social_edges_user0']
        self.social_edges_user1 = data_dict['social_edges_user1']
        self.social_neighbors_indices_input = np.concatenate([self.social_edges_user0, self.social_edges_user1], 1)
        self.social_neighbors_dense_shape = np.array([self.conf.num_users, self.conf.num_users]).astype(np.int64)
        self.seg_idx = tf.constant(np.squeeze(self.social_edges_user0), dtype=tf.int32)

    def sparse_matrix_values(self, sp_value):
        tmp = tf.SparseTensor(
            indices = self.social_neighbors_indices_input,
            values = tf.squeeze(sp_value),
            dense_shape=self.social_neighbors_dense_shape
        )
        return tf.expand_dims(tf.sparse.softmax(tmp)._values, 1)

    def initializeNodes(self):
        super(samn, self).initializeNodes()
        self.K = tf.Variable(tf.random_normal([self.dim, self.conf.num_mem], mean=0.0, stddev=0.01), name='K')
        self.M = tf.Variable(tf.random_normal([self.conf.num_mem, self.dim], mean=0.0, stddev=0.01), name='M')
        self.W = tf.layers.Dense(self.conf.att_dim, activation=tf.nn.leaky_relu, name='W', use_bias=True)
        self.h = tf.layers.Dense(1, name='h', use_bias=True)

    def constructTrainGraph(self):
        u0 = tf.math.l2_normalize(tf.gather_nd(self.user_embedding, self.social_edges_user0), axis=1)
        u1 = tf.math.l2_normalize(tf.gather_nd(self.user_embedding, self.social_edges_user1), axis=1)
        f_ul = tf.matmul(tf.nn.softmax(tf.matmul(u0*u1, self.K)), self.M)
        sp_value = self.h(self.W(f_ul))
        sp_value = self.sparse_matrix_values(sp_value)
        user_embedding_last = self.user_embedding + tf.unsorted_segment_sum(u1*f_ul*sp_value, self.seg_idx, num_segments=self.conf.num_users)
        item_embedding_last = self.item_embedding
        latest_user_embedding_latent = tf.gather_nd(user_embedding_last, self.user_input)
        latest_item_latent = tf.gather_nd(item_embedding_last, self.item_input)
        latest_item_neg_latent = tf.gather_nd(item_embedding_last, self.item_neg_input)

        self.prediction = tf.sigmoid(tf.matmul(latest_user_embedding_latent, tf.transpose(item_embedding_last)))
        W_loss = tf.add_n([tf.reduce_sum(tf.square(self.W.kernel)), tf.reduce_sum(tf.square(self.W.bias)), \
            tf.reduce_sum(tf.square(self.h.kernel)), tf.reduce_sum(tf.square(self.h.bias))])
        reg_loss = tf.add_n([tf.reduce_sum(tf.square(latest_user_embedding_latent)), tf.reduce_sum(tf.square(latest_item_latent)), tf.reduce_sum(tf.square(latest_item_neg_latent)), W_loss])
        interaction_BPR_vector = tf.multiply(latest_user_embedding_latent, latest_item_latent - latest_item_neg_latent)
        BPR_loss = -tf.reduce_sum(tf.log(tf.sigmoid(tf.reduce_sum(interaction_BPR_vector, 1, keepdims=True))))
        self.loss = self.reg*reg_loss + BPR_loss
        self.Adam = tf.train.AdamOptimizer(self.learning_rate)
        self.opt = self.Adam.minimize(self.loss)
        self.init = tf.global_variables_initializer()

    def saveVariables(self):
        variables_dict = {}
        variables_dict['user_embedding'] = self.user_embedding
        variables_dict['item_embedding'] = self.item_embedding
        variables_dict['K'] = self.K
        variables_dict['M'] = self.M
        variables_dict['W/kernel'] = self.W.kernel
        variables_dict['W/bias'] = self.W.bias
        variables_dict['h/kernel'] = self.h.kernel
        variables_dict['h/bias'] = self.h.bias
        variables_dict['Adam'] = self.Adam
        self.saver = tf.train.Saver(variables_dict)