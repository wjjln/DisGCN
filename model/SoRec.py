import tensorflow as tf
import tensorflow.compat.v1 as tfv1
import numpy as np
import os
from base_model import base_model

class sorec(base_model):
    def inputSupply(self, data_dict):
        self.social_edges_user0 = np.squeeze(data_dict['social_edges_user0'], -1)
        self.social_edges_user1 = np.squeeze(data_dict['social_edges_user1'], -1)
        self.social_edges_user_neg = data_dict['social_edges_user_neg']

    def initializeNodes(self):
        super(sorec, self).initializeNodes()
        self.user_social_loss_idx = tfv1.placeholder("int32", [None])
        self.feat_embedding = tf.Variable(
                tf.random.normal([self.conf.num_users, self.dim], stddev=0.01), name='feat_embedding')

    def constructTrainGraph(self):
        u0 = tf.gather(self.social_edges_user0, self.user_social_loss_idx)
        u1 = tf.gather(self.social_edges_user1, self.user_social_loss_idx)
        un = tf.gather(self.social_edges_user_neg, self.user_social_loss_idx)
        u_emb = tf.gather(self.user_embedding, u0)
        f_emb = tf.gather(self.feat_embedding, u1)
        f_emb_n = tf.gather_nd(self.feat_embedding, un)
        BPR_vector = tf.multiply(tf.expand_dims(u_emb, 1), -(tf.expand_dims(f_emb, 1)-f_emb_n))
        social_loss = tf.reduce_sum(tf.reduce_mean(tf.nn.softplus(tf.reduce_sum(BPR_vector, -1)), -1))
        self.social_loss = social_loss + self.reg*(self.regloss([u_emb, f_emb]) + self.regloss([f_emb_n])/self.num_negatives)
        self.predict()

        self.loss = self.BPRloss() + self.social_loss
        self.opt = tfv1.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        self.init = tfv1.global_variables_initializer()

    def saveVariables(self):
        variables_dict = {}
        variables_dict['user_embedding'] = self.user_embedding
        variables_dict['item_embedding'] = self.item_embedding
        variables_dict['feat_embedding'] = self.feat_embedding
        self.saver = tfv1.train.Saver(variables_dict)
