import tensorflow as tf
import numpy as np
import os
from base_model import base_model

class Geobpr(base_model):
    def inputSupply(self, data_dict):
        self.alpha = self.conf.alpha
        interaction_user_indices = data_dict['interaction_user_indices']
        interaction_item_indices = data_dict['interaction_item_indices']
        consumed_items_indices_input = np.concatenate([interaction_user_indices, interaction_item_indices], 1)
        consumed_items_values_input = data_dict['consumed_items_values_input']
        consumed_items_dense_shape = np.array([self.conf.num_users, self.conf.num_items]).astype(np.int64)
        self.consumed_items_sparse_matrix = tf.SparseTensor(
            indices = consumed_items_indices_input, 
            values = consumed_items_values_input,
            dense_shape=consumed_items_dense_shape
        )

    def initializeNodes(self):
        super(Geobpr, self).initializeNodes()
        self.Geo0_input = tf.placeholder("int32", [None, 1])
        self.Geo1_input = tf.placeholder("int32", [None, 1])
        # self.M = tf.Variable(tf.random_normal([self.conf.num_mem, self.conf.geo_dim], stddev=0.01), name='M')
        # self.K = tf.Variable(tf.random_normal([self.conf.num_mem, self.conf.dimension], stddev=0.01), name='K')

    def constructTrainGraph(self):
        # Geo_iemb = tf.matmul(tf.nn.softmax(tf.matmul(self.item_embedding, tf.transpose(self.K))), self.M)
        Geo_iemb = tf.Variable(tf.random_normal([self.conf.num_items, self.conf.dimension], stddev=0.01), name='Geo_iemb')
        Geo_uemb = tf.sparse_tensor_dense_matmul(self.consumed_items_sparse_matrix, Geo_iemb)
        u_emb = tf.gather_nd(self.user_embedding, self.user_input)
        i_emb = tf.gather_nd(self.item_embedding, self.item_input)
        i_emb_n = tf.gather_nd(self.item_embedding, self.item_neg_input)
        Geo_u = tf.gather_nd(Geo_uemb, self.user_input)
        Geo_i = tf.gather_nd(Geo_iemb, self.item_input)
        Geo_i_n = tf.gather_nd(Geo_iemb, self.item_neg_input)
        Geo0 = tf.gather_nd(Geo_iemb, self.Geo0_input)
        Geo1 = tf.gather_nd(Geo_iemb, self.Geo1_input)
        score_p = tf.reduce_sum(u_emb*i_emb, 1) + self.alpha*tf.reduce_sum(Geo_u*Geo_i, 1)
        score_n = tf.reduce_sum(u_emb*i_emb_n, 1) + self.alpha*tf.reduce_sum(Geo_u*Geo_i_n, 1)
        Geo_p = tf.reduce_sum(Geo_i*Geo0, 1)
        Geo_n= tf.reduce_sum(Geo_i*Geo1, 1)
        self.prediction = tf.sigmoid(tf.matmul(u_emb, tf.transpose(self.item_embedding)) + self.alpha*tf.matmul(Geo_u, tf.transpose(Geo_iemb)))
        self.reg_loss = tf.add_n([tf.nn.l2_loss(u_emb), tf.nn.l2_loss(i_emb), tf.nn.l2_loss(i_emb_n), tf.nn.l2_loss(Geo_u), tf.nn.l2_loss(Geo_i),\
            tf.nn.l2_loss(Geo_i_n), tf.nn.l2_loss(Geo0), tf.nn.l2_loss(Geo1)])
        # self.reg_loss += tf.add_n([tf.nn.l2_loss(self.M), tf.nn.l2_loss(self.K)])
        self.loss = self.reg*self.reg_loss\
                    - tf.reduce_sum(tf.log(tf.sigmoid(score_p - score_n))) - self.alpha*tf.reduce_sum(tf.log(tf.sigmoid(Geo_p - Geo_n)))
        self.opt = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        self.init = tf.global_variables_initializer()

    def saveVariables(self):
        variables_dict = {}
        variables_dict['user_embedding'] = self.user_embedding
        variables_dict['item_embedding'] = self.item_embedding
        # variables_dict['K'] = self.K
        # variables_dict['M'] = self.M
        self.saver = tf.train.Saver(variables_dict)

    def defineMap(self):
        super(Geobpr, self).defineMap()
        self.map_dict['train'][self.Geo0_input] = 'Geo0_list'
        self.map_dict['test'][self.Geo0_input] = 'Geo0_list'
        self.map_dict['val'][self.Geo0_input] = 'Geo0_list'
        self.map_dict['train'][self.Geo1_input] = 'Geo1_list'
        self.map_dict['test'][self.Geo1_input] = 'Geo1_list'
        self.map_dict['val'][self.Geo1_input] = 'Geo1_list'
