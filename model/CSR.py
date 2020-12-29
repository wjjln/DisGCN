import tensorflow as tf
import tensorflow.compat.v1 as tfv1
import numpy as np
import os
from base_model import base_model

class csr(base_model):
    def inputSupply(self, data_dict):
        self.u_input = data_dict['u_input']
        self.f_input = data_dict['f_input']
        self.i_input = data_dict['i_input']
        self.seg_in_friends = data_dict['seg_in_friends']
        self.u_idx = data_dict['u_idx']

    def initializeNodes(self):
        super(csr, self).initializeNodes()

    def constructTrainGraph(self):
        loss = self.BPRloss()
        u_emb, f_emb, i_emb = tf.gather(self.user_embedding, self.f_input), tf.gather(self.user_embedding, self.f_input), tf.gather(self.item_embedding, self.i_input)
        ufi_csr = tf.square((u_emb-f_emb)*i_emb)
        uf_csr = tf.math.segment_sum(ufi_csr, self.seg_in_friends)
        u_csr = tf.math.unsorted_segment_sum(uf_csr, self.u_idx, self.num_users)
        self.loss = loss + self.reg*tf.reduce_sum(tf.gather(u_csr, self.user_input))
        self.predict()
        self.opt = tfv1.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        self.init = tfv1.global_variables_initializer()

    def saveVariables(self):
        variables_dict = {}
        variables_dict['user_embedding'] = self.user_embedding
        variables_dict['item_embedding'] = self.item_embedding
        self.saver = tfv1.train.Saver(variables_dict)
