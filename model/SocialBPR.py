import tensorflow as tf
import tensorflow.compat.v1 as tfv1
import numpy as np
import os
from base_model import base_model

class socialbpr(base_model):
    def inputSupply(self, data_dict):
        self.social_edges_user0 = data_dict['social_edges_user0']
        self.social_edges_user1 = data_dict['social_edges_user1']
        self.social_neighbors_indices_input = np.concatenate([self.social_edges_user0, self.social_edges_user1], 1)
        self.social_neighbors_values_input = data_dict['social_neighbors_values_input']
        self.social_neighbors_dense_shape = np.array([self.conf.num_users, self.conf.num_users]).astype(np.int64)

    def initializeNodes(self):
        super(socialbpr, self).initializeNodes()

    def constructTrainGraph(self):
        social_neighbors_sparse_matrix = tf.SparseTensor(
            indices = self.social_neighbors_indices_input, 
            values = self.social_neighbors_values_input,
            dense_shape=self.social_neighbors_dense_shape
        )
        user_emb_from_friends = tf.sparse.sparse_dense_matmul(social_neighbors_sparse_matrix, self.user_embedding)
        self.loss = self.BPRloss() + self.reg*self.regloss([tf.gather(self.user_embedding, self.user_input) - tf.gather(user_emb_from_friends, self.user_input)])
        self.predict()
        self.opt = tfv1.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        self.init = tfv1.global_variables_initializer()

    def saveVariables(self):
        variables_dict = {}
        variables_dict['user_embedding'] = self.user_embedding
        variables_dict['item_embedding'] = self.item_embedding
        self.saver = tfv1.train.Saver(variables_dict)
