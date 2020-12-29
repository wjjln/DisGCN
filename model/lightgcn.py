import tensorflow as tf
import numpy as np
import os
from base_model import base_model

class lightgcn(base_model):
    def inputSupply(self, data_dict):
        self.dp = tf.placeholder("float32", [1])
        self.dp_layer = tf.layers.Dropout(tf.gather(self.dp, 0))
        self.social_edges_user0 = data_dict['social_edges_user0']
        self.social_edges_user1 = data_dict['social_edges_user1']
        self.interaction_user_indices = data_dict['interaction_user_indices']
        self.interaction_item_indices = data_dict['interaction_item_indices']
        self.user_self_value = data_dict['user_self_value_CSR']
        self.item_self_value = data_dict['item_self_value_CSR']
        self.consumed_items_indices_input = np.concatenate([self.interaction_user_indices, self.interaction_item_indices], 1)
        self.consumed_items_values_input = data_dict['consumed_items_values_input_CSR']
        self.social_neighbors_indices_input = np.concatenate([self.social_edges_user0, self.social_edges_user1], 1)
        self.social_neighbors_values_input = data_dict['social_neighbors_values_input_CSR']
        self.social_neighbors_values_input = self.dp_layer(self.social_neighbors_values_input)
        self.consumed_items_values_input = self.dp_layer(self.consumed_items_values_input)
        self.inter_users_indices_input = np.concatenate([self.interaction_item_indices, self.interaction_user_indices], 1)
        self.inter_users_values_input = data_dict['inter_users_values_input_CSR']
        self.inter_users_values_input = self.dp_layer(self.inter_users_values_input)
        self.consumed_items_dense_shape = np.array([self.conf.num_users, self.conf.num_items]).astype(np.int64)
        self.social_neighbors_dense_shape = np.array([self.conf.num_users, self.conf.num_users]).astype(np.int64)
        self.inter_users_dense_shape = np.array([self.conf.num_items, self.conf.num_users]).astype(np.int64)

        self.social_neighbors_sparse_matrix = self.construct_social_neighbors_sparse_matrix(self.social_neighbors_values_input)
        self.consumed_items_sparse_matrix = self.construct_consumed_items_sparse_matrix(self.consumed_items_values_input)
        self.inter_users_sparse_matrix = self.construct_inter_users_sparse_matrix(self.inter_users_values_input)

    def initializeNodes(self):
        super(lightgcn, self).initializeNodes()
        self.num_layers = self.conf.num_layers
        # self.u_self_connect = [tf.layers.Dense(self.conf.dimension, activation=tf.nn.leaky_relu, name='u_self_connect', use_bias=True) for _ in range(self.num_layers)]
        # self.i_self_connect = [tf.layers.Dense(self.conf.dimension, activation=tf.nn.leaky_relu, name='i_self_connect', use_bias=True) for _ in range(self.num_layers)]
        
    def propagate(self, user_p, layer, item_e=None):
        user_embedding_from_item = tf.sparse_tensor_dense_matmul(self.consumed_items_sparse_matrix, item_e)
        user_embedding_from_friend = tf.sparse_tensor_dense_matmul(self.social_neighbors_sparse_matrix, user_p)
        item_embedding_from_user = tf.sparse_tensor_dense_matmul(self.inter_users_sparse_matrix, user_p)
        user_embedding_last = user_embedding_from_friend + user_embedding_from_item + self.user_self_value*user_p
        item_embedding_last = item_embedding_from_user + self.item_self_value*item_e
        # user_embedding_last = self.u_self_connect[layer](user_embedding_from_friend + user_embedding_from_item + self.user_self_value*user_p)
        # item_embedding_last = self.i_self_connect[layer](item_embedding_from_user + self.item_self_value*item_e)
        return user_embedding_last, item_embedding_last 

    def constructTrainGraph(self):
        user_embedding_last, item_embedding_last = self.user_embedding, self.item_embedding
        # tmp_user_embedding, tmp_item_embedding = self.user_embedding, self.item_embedding
        # [tf.math.l2_normalize(tmp_user_embedding, axis=1)], [tf.math.l2_normalize(tmp_item_embedding, axis=1)]
        for k in range(self.num_layers):
            user_embedding_last, item_embedding_last = self.propagate(user_embedding_last, k, item_embedding_last)
            # user_embedding_last.append(tf.math.l2_normalize(tmp_user_embedding, axis=1))
            # item_embedding_last.append(tf.math.l2_normalize(tmp_item_embedding, axis=1))
        # user_embedding_last = tf.concat(user_embedding_last, 1)
        # item_embedding_last = tf.concat(item_embedding_last, 1)
        latest_user_embedding_latent = tf.gather_nd(user_embedding_last, self.user_input)
        latest_item_latent = tf.gather_nd(item_embedding_last, self.item_input)
        latest_item_neg_latent = tf.gather_nd(item_embedding_last, self.item_neg_input)
        # W_loss = 0
        # for k in range(self.num_layers):
        #     W_loss += (tf.reduce_sum(tf.square(self.u_self_connect[k].kernel)) + tf.reduce_sum(tf.square(self.u_self_connect[k].bias)))
        #     W_loss += (tf.reduce_sum(tf.square(self.i_self_connect[k].kernel)) + tf.reduce_sum(tf.square(self.i_self_connect[k].bias)))
        interaction_BPR_vector = tf.multiply(latest_user_embedding_latent, latest_item_latent - latest_item_neg_latent)
        self.prediction = tf.sigmoid(tf.matmul(latest_user_embedding_latent, tf.transpose(item_embedding_last)))
        self.reg_loss = tf.add_n([tf.reduce_sum(tf.square(tf.gather_nd(user_embedding_last, self.unique_user_list))), tf.reduce_sum(tf.square(tf.gather_nd(item_embedding_last, self.unique_item_list))), tf.reduce_sum(tf.square(tf.gather_nd(item_embedding_last, self.unique_item_neg_list)))])
        self.BPR_loss = tf.reduce_sum(tf.log(tf.sigmoid(tf.reduce_sum(interaction_BPR_vector, 1, keepdims=True))))
        self.loss = self.reg * self.reg_loss - self.BPR_loss
        self.opt = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        self.init = tf.global_variables_initializer()

    def saveVariables(self):
        variables_dict = {}
        variables_dict['user_embedding'] = self.user_embedding
        variables_dict['item_embedding'] = self.item_embedding
        self.saver = tf.train.Saver(variables_dict)

    def construct_social_neighbors_sparse_matrix(self, sp_value):
        return tf.SparseTensor(
            indices = self.social_neighbors_indices_input,
            values = sp_value,
            dense_shape=self.social_neighbors_dense_shape
        )
    def construct_consumed_items_sparse_matrix(self, sp_value):
        return tf.SparseTensor(
            indices = self.consumed_items_indices_input, 
            values = sp_value,
            dense_shape=self.consumed_items_dense_shape
        )
    def construct_inter_users_sparse_matrix(self, sp_value):
        return tf.SparseTensor(
            indices = self.inter_users_indices_input, 
            values = sp_value,
            dense_shape=self.inter_users_dense_shape
        )