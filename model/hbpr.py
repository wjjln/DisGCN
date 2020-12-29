import tensorflow as tf
import numpy as np
import os
from base_model import base_model

class hbpr(base_model):
    def inputSupply(self, data_dict):
        pass

    def initializeNodes(self):
        super(hbpr, self).initializeNodes()
        self.relation_mean = tf.Variable(tf.random_normal([self.conf.num_mem, self.conf.dimension], mean=1.0, stddev=0.01), name='relation_mean')
        self.relation_var = tf.Variable(tf.random_normal([self.conf.num_mem, self.conf.dimension], stddev=0.01), name='relation_var')
        self.m = tf.distributions.Normal(0.0, 1.0)
        self.W_z = [tf.layers.Dense(100, activation=tf.nn.leaky_relu, name='W_z', use_bias=True), tf.layers.Dense(self.conf.num_mem, name='W_z', use_bias=True)]

    def edge_gen(self, user_emb, item_emb, user, item):
        u_emb = tf.gather_nd(user_emb, user)
        i_emb = tf.gather_nd(item_emb, item)
        edges_preference_cat = self.W_z[1](self.W_z[0](tf.concat([u_emb, i_emb, u_emb*i_emb], 1)))
        edges_preference_cat_softmax = tf.nn.softmax(edges_preference_cat)
        m_sample = self.m.sample([self.conf.num_mem, self.conf.dimension])
        M = self.relation_mean + self.relation_var*m_sample
        edges_preference = tf.matmul(edges_preference_cat_softmax, M)
        return edges_preference

    def constructTrainGraph(self):
        u_emb = tf.gather_nd(self.user_embedding, self.user_input)
        i_emb = tf.gather_nd(self.item_embedding, self.item_input)
        i_emb_n = tf.gather_nd(self.item_embedding, self.item_neg_input)
        edges_preference = self.edge_gen(self.user_embedding, self.item_embedding, self.user_input, self.item_input)
        edges_preference_neg = self.edge_gen(self.user_embedding, self.item_embedding, self.user_input, self.item_neg_input)
        edge_pos = tf.multiply(tf.multiply(u_emb, i_emb), edges_preference)
        edge_neg = tf.multiply(tf.multiply(u_emb, i_emb_n), edges_preference_neg)
        BPR_vector = edge_pos - edge_neg
        self.prediction = tf.sigmoid(tf.reduce_sum(edge_pos, 1))
        W_loss = 0
        W_loss += (tf.reduce_sum(tf.square(self.W_z[0].kernel)) + tf.reduce_sum(tf.square(self.W_z[0].bias))\
                                + tf.reduce_sum(tf.square(self.W_z[1].kernel)) + tf.reduce_sum(tf.square(self.W_z[1].bias)))
        W_loss += tf.reduce_sum(tf.square(self.relation_mean))
        W_loss += tf.reduce_sum(tf.square(self.relation_var))
        self.loss = self.reg*tf.add_n([tf.reduce_sum(tf.square(tf.gather_nd(self.user_embedding, self.unique_user_list))), tf.reduce_sum(tf.square(tf.gather_nd(self.item_embedding, self.unique_item_list))), tf.reduce_sum(tf.square(tf.gather_nd(self.item_embedding, self.unique_item_neg_list))), W_loss])\
                    - tf.reduce_sum(tf.log(tf.sigmoid(tf.reduce_sum(BPR_vector, axis=1))))
        self.opt = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        self.init = tf.global_variables_initializer()

    def saveVariables(self):
        variables_dict_emb = {}
        variables_dict_emb['user_embedding'] = self.user_embedding
        variables_dict_emb['item_embedding'] = self.item_embedding
        variables_dict_mem = {}
        variables_dict_mem['relation_mean'] = self.relation_mean
        variables_dict_mem['relation_var'] = self.relation_var
        variables_dict_mem['W_z/kernel'] = self.W_z[0].kernel
        variables_dict_mem['W_z/bias'] = self.W_z[0].bias
        variables_dict_mem['W_z_1/kernel'] = self.W_z[1].kernel
        variables_dict_mem['W_z_1/bias'] = self.W_z[1].bias
        self.saver_emb = tf.train.Saver(variables_dict_emb)
        self.saver_mem = tf.train.Saver(variables_dict_mem)
