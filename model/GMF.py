import tensorflow as tf
import numpy as np
import os
from base_model import base_model

class gmf(base_model):
    def inputSupply(self, data_dict):
        pass

    def initializeNodes(self):
        self.item_input = tf.placeholder("int32", [None, 1])
        self.user_input = tf.placeholder("int32", [None, 1])
        self.item_neg_input = tf.placeholder("int32", [None, 1])

        self.user_embedding_GMF = tf.Variable(
            tf.random_normal([self.conf.num_users, self.conf.dimension], stddev=0.01), name='user_embedding_GMF')
        self.item_embedding_GMF = tf.Variable(
            tf.random_normal([self.conf.num_items, self.conf.dimension], stddev=0.01), name='item_embedding_GMF')

        self.h_GMF = tf.layers.Dense(\
                1, name='h_GMF', use_bias=False)

        self.emb_loss = 0

    def inference(self, item_input):
        i_emb_GMF = tf.gather_nd(self.item_embedding_GMF, item_input)
        self.emb_loss += tf.reduce_sum(tf.square(i_emb_GMF))
        emb_GMF = self.u_emb_GMF*i_emb_GMF
        return self.h_GMF(emb_GMF)

    def constructTrainGraph(self):
        self.u_emb_GMF = tf.gather_nd(self.user_embedding_GMF, self.user_input)
        self.emb_loss += tf.reduce_sum(tf.square(self.u_emb_GMF))
        score_p = self.inference(self.item_input)
        score_n = self.inference(self.item_neg_input)
        self.prediction = score_p
        W_loss = tf.reduce_sum(tf.square(self.h_GMF.kernel))
        # self.loss = -tf.reduce_sum(tf.log(score_p) + tf.log(1-score_n)) + self.reg*self.reg_loss
        self.loss = -tf.reduce_sum(tf.log(tf.sigmoid(score_p-score_n))) + self.reg*self.emb_loss + self.conf.w_reg*W_loss
        self.opt = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        self.init = tf.global_variables_initializer()

    def saveVariables(self):
        variables_dict_GMF = {}
        variables_dict_GMF['user_embedding_GMF'] = self.user_embedding_GMF
        variables_dict_GMF['item_embedding_GMF'] = self.item_embedding_GMF
        variables_dict_GMF['h_GMF/kernel'] = self.h_GMF.kernel
        self.saver = tf.train.Saver(variables_dict_GMF)