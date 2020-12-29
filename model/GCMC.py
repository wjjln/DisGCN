import tensorflow as tf
import numpy as np
import os
from base_model import base_model

class gcmc(base_model):
    def inputSupply(self, data_dict):
        self.interaction_user_indices = data_dict['interaction_user_indices']
        self.interaction_item_indices = data_dict['interaction_item_indices']
        self.consumed_items_indices_input = np.concatenate([self.interaction_user_indices, self.interaction_item_indices], 1)
        self.consumed_items_values_input = data_dict['consumed_items_values_input']
        self.inter_users_indices_input = np.concatenate([self.interaction_item_indices, self.interaction_user_indices], 1)
        self.inter_users_values_input = data_dict['inter_users_values_input']
        self.consumed_items_dense_shape = np.array([self.conf.num_users, self.conf.num_items]).astype(np.int64)
        self.inter_users_dense_shape = np.array([self.conf.num_items, self.conf.num_users]).astype(np.int64)

        self.consumed_items_sparse_matrix = self.construct_consumed_items_sparse_matrix(self.consumed_items_values_input)
        self.inter_users_sparse_matrix = self.construct_inter_users_sparse_matrix(self.inter_users_values_input)

        self.user_side = tf.constant(data_dict['user_side_information'], dtype=tf.float32)

    def initializeNodes(self):
        super(gcmc, self).initializeNodes()
        self.W1 = tf.layers.Dense(self.dim, activation=tf.nn.leaky_relu, name='W1', use_bias=True)
        self.W = tf.layers.Dense(self.dim, activation=tf.nn.leaky_relu, name='W', use_bias=True)
        self.Wh = tf.layers.Dense(self.dim, name='Wh', use_bias=True)
        self.Wf = tf.layers.Dense(self.dim, name='Wf', use_bias=True)

    def constructTrainGraph(self):
        user_embedding_last = self.W1(tf.sparse_tensor_dense_matmul(self.consumed_items_sparse_matrix, self.item_embedding))
        item_embedding_last = self.W1(tf.sparse_tensor_dense_matmul(self.inter_users_sparse_matrix, self.user_embedding))
        user_embedding_last = tf.nn.leaky_relu(self.Wh(user_embedding_last) + self.Wf(self.W(self.user_side)))

        latest_user_embedding_latent = tf.gather_nd(user_embedding_last, self.user_input)
        latest_item_latent = tf.gather_nd(item_embedding_last, self.item_input)
        latest_item_neg_latent = tf.gather_nd(item_embedding_last, self.item_neg_input)
        W_loss = 0
        W_loss += (tf.reduce_sum(tf.square(self.W1.kernel)) + tf.reduce_sum(tf.square(self.W1.bias)))
        W_loss += (tf.reduce_sum(tf.square(self.W.kernel)) + tf.reduce_sum(tf.square(self.W.bias)))
        W_loss += (tf.reduce_sum(tf.square(self.Wh.kernel)) + tf.reduce_sum(tf.square(self.Wh.bias)))
        W_loss += (tf.reduce_sum(tf.square(self.Wf.kernel)) + tf.reduce_sum(tf.square(self.Wf.bias)))

        interaction_BPR_vector = tf.multiply(latest_user_embedding_latent, latest_item_latent - latest_item_neg_latent)
        self.prediction = tf.sigmoid(tf.matmul(latest_user_embedding_latent, tf.transpose(item_embedding_last)))
        self.reg_loss = tf.add_n([tf.reduce_sum(tf.square(latest_user_embedding_latent)), tf.reduce_sum(tf.square(latest_item_latent)), tf.reduce_sum(tf.square(latest_item_neg_latent)), W_loss])
        self.BPR_loss = tf.reduce_sum(tf.log(tf.sigmoid(tf.reduce_sum(interaction_BPR_vector, 1, keepdims=True))))
        self.loss = self.reg * self.reg_loss - self.BPR_loss
        self.opt = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        self.init = tf.global_variables_initializer()

    def saveVariables(self):
        variables_dict = {}
        variables_dict['user_embedding'] = self.user_embedding
        variables_dict['item_embedding'] = self.item_embedding
        variables_dict['W1/kernel'] = self.W1.kernel
        variables_dict['W1/bias'] = self.W1.bias
        variables_dict['Wh/kernel'] = self.Wh.kernel
        variables_dict['Wh/bias'] = self.Wh.bias
        variables_dict['W/kernel'] = self.W.kernel
        variables_dict['W/bias'] = self.W.bias
        variables_dict['Wf/kernel'] = self.Wf.kernel
        variables_dict['Wf/bias'] = self.Wf.bias
        self.saver = tf.train.Saver(variables_dict)

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