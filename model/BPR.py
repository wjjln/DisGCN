import tensorflow as tf
import tensorflow.compat.v1 as tfv1
import numpy as np
import os
from base_model import base_model

class bpr(base_model):
    # def inputSupply(self, data_dict):
    #     social_edges_user0 = data_dict['social_edges_user0']
    #     social_edges_user1 = data_dict['social_edges_user1']
    #     self.social_neighbors_indices_input = np.concatenate([social_edges_user0, social_edges_user1], 1)
    #     self.social_neighbors_values_input = data_dict['social_neighbors_values_input']
    #     self.social_neighbors_dense_shape = np.array([self.num_users, self.num_users]).astype(np.int64)

    def initializeNodes(self):
        super(bpr, self).initializeNodes()

    def constructTrainGraph(self):
        # social_neighbors_sparse_matrix = tf.SparseTensor(
        #     indices = self.social_neighbors_indices_input, 
        #     values = self.social_neighbors_values_input,
        #     dense_shape=self.social_neighbors_dense_shape
        # )
        # from_friend = tf.sparse.sparse_dense_matmul(social_neighbors_sparse_matrix, self.user_embedding)
        # pred = self.conf.pred
        # if pred == 0:
        self.prediction = self.predict()
        # elif pred == 1:
        #     self.predict(from_friend)
        # elif pred == 2:
        #     self.predict(from_friend+self.user_embedding)
        self.loss = self.BPRloss()
        self.opt = tfv1.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        self.init = tfv1.global_variables_initializer()
        

    def saveVariables(self):
        variables_dict = {}
        variables_dict['user_embedding'] = self.user_embedding
        variables_dict['item_embedding'] = self.item_embedding
        self.saver = tfv1.train.Saver(variables_dict)
