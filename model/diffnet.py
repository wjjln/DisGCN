
import tensorflow as tf
import tensorflow.compat.v1 as tfv1
import numpy as np
import os
from base_model import base_model

class diffnet(base_model):

    def inputSupply(self, data_dict):
        social_edges_user0 = data_dict['social_edges_user0']
        social_edges_user1 = data_dict['social_edges_user1']
        interaction_user_indices = data_dict['interaction_user_indices']
        interaction_item_indices = data_dict['interaction_item_indices']

        self.social_neighbors_indices_input = np.concatenate([social_edges_user0, social_edges_user1], 1)
        self.social_neighbors_values_input = data_dict['social_neighbors_values_input']
        self.consumed_items_indices_input = np.concatenate([interaction_user_indices, interaction_item_indices], 1)
        self.consumed_items_values_input = data_dict['consumed_items_values_input']
        self.social_neighbors_dense_shape = np.array([self.conf.num_users, self.conf.num_users]).astype(np.int64)
        self.consumed_items_dense_shape = np.array([self.conf.num_users, self.conf.num_items]).astype(np.int64)

    # def convertDistribution(self, x):
    #     mean, var = tf.nn.moments(x, axes=[0, 1])
    #     y = (x - mean) * 0.1 / tf.sqrt(var)
    #     return y

    def initializeNodes(self):
        super(diffnet, self).initializeNodes()
        # regularizer = tfv1.contrib.layers.l2_regularizer(self.reg)
        self.graph_SAGElayer = [tfv1.layers.Dense(self.conf.dimension, activation=tf.nn.leaky_relu, name=f'graph_SAGElayer_{k}', use_bias=True) for k in range(self.conf.num_layers)]

    def constructTrainGraph(self):
        social_neighbors_sparse_matrix = tf.SparseTensor(
            indices = self.social_neighbors_indices_input, 
            values = self.social_neighbors_values_input,
            dense_shape = self.social_neighbors_dense_shape
        )
        consumed_items_sparse_matrix = tf.SparseTensor(
            indices = self.consumed_items_indices_input, 
            values = self.consumed_items_values_input,
            dense_shape = self.consumed_items_dense_shape
        )

        u_emb = self.user_embedding
        # user_embedding_last_list = []
        user_embedding_from_consumed_items = tf.sparse.sparse_dense_matmul(consumed_items_sparse_matrix, self.item_embedding)
        for k in range(self.conf.num_layers):
            user_embedding_from_user_embedding = tf.sparse.sparse_dense_matmul(social_neighbors_sparse_matrix, u_emb)
            concat_user_embedding = tf.concat([user_embedding_from_user_embedding, u_emb], 1) # concat [mean of friends emb, self-loop user emb]
            u_emb = self.graph_SAGElayer[k](concat_user_embedding)
            # user_embedding_last_list.append(u_emb)
            # u_emb = self.convertDistribution(u_emb)
        self.u_emb = user_embedding_from_consumed_items + u_emb
        # + tf.add_n(user_embedding_last_list)

        # self.prediction = self.predict(self.u_emb)
        self.prediction = self.predict(tf.transpose(tf.gather(tf.transpose(self.u_emb), tf.range(32))), tf.transpose(tf.gather(tf.transpose(self.item_embedding), tf.range(32))))
        self.loss = self.BPRloss(self.u_emb)
        # + tf.losses.get_regularization_loss()
        self.Adam = tfv1.train.AdamOptimizer(self.learning_rate)
        self.opt = self.Adam.minimize(self.loss)
        self.init = tfv1.global_variables_initializer()

    def saveVariables(self):
        variables_dict = {}
        variables_dict['user_embedding'] = self.user_embedding
        variables_dict['item_embedding'] = self.item_embedding
        for k in range(self.conf.num_layers):
            variables_dict['graph_SAGElayer_{}/kernel'.format(k)] = self.graph_SAGElayer[k].kernel
            variables_dict['graph_SAGElayer_{}/bias'.format(k)] = self.graph_SAGElayer[k].bias

        variables_dict['Adam'] = self.Adam
        self.saver = tfv1.train.Saver(variables_dict)
        variables_dict_emb = {}
        variables_dict_emb['user_embedding'] = self.user_embedding
        variables_dict_emb['item_embedding'] = self.item_embedding
        self.saver_emb = tfv1.train.Saver(variables_dict_emb)
        