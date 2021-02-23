import tensorflow as tf
import tensorflow.compat.v1 as tfv1
import numpy as np
import os
from base_model import base_model

class ngcf(base_model):

    def inputSupply(self, data_dict):
        self.num_users = self.conf.num_users
        self.num_items = self.conf.num_items
        interaction_user_indices = data_dict['interaction_user_indices']
        interaction_item_indices = data_dict['interaction_item_indices']
        self.consumed_items_values_input= data_dict['consumed_items_values_input']
        self.inter_users_values_input = data_dict['inter_users_values_input']
        self.user_self_value = data_dict['user_self_value']
        self.item_self_value = data_dict['item_self_value']
        self.consumed_items_indices_input = np.concatenate([interaction_user_indices, interaction_item_indices], 1)
        self.consumed_items_dense_shape = np.array([self.num_users, self.num_items]).astype(np.int64)
        self.inter_users_indices_input = np.concatenate([interaction_item_indices, interaction_user_indices], 1)
        self.inter_users_dense_shape = np.array([self.num_items, self.num_users]).astype(np.int64)
        # self.user_side = tf.constant(data_dict['user_side_information'], dtype=tf.float32)
        # t = np.reshape(range(self.num_users), [-1, 1])
        # self_users_indices_input = np.concatenate([t, t], 1)
        # t = np.reshape(range(self.num_items), [-1, 1])
        # self_items_indices_input = np.concatenate([t, t], 1)
        # self_users_shape = np.array([self.num_users, self.num_users]).astype(np.int64)
        # self_items_shape = np.array([self.num_items, self.num_items]).astype(np.int64)
        # self.L_I = tf.sparse.concat(0, [tf.sparse.concat(1, [self.self_user_sparse_matrix, self.consumed_items_sparse_matrix]), \
        #     tf.sparse.concat(1, [self.self_item_sparse_matrix, self.inter_users_sparse_matrix])])
    def construct_consumed_items_sparse_matrix(self, sp_value):
        return tf.SparseTensor(
            indices = self.consumed_items_indices_input, 
            values = tf.squeeze(sp_value),
            dense_shape=self.consumed_items_dense_shape
        )
    
    def construct_interacted_users_sparse_matrix(self, sp_value):
        return tf.SparseTensor(
            indices = self.inter_users_indices_input, 
            values = tf.squeeze(sp_value),
            dense_shape=self.inter_users_dense_shape
        )

    def initializeNodes(self):
        super(ngcf, self).initializeNodes()
        self.training = tfv1.placeholder(tf.bool, ())
        self.node_dropout = tfv1.layers.Dropout(self.conf.node_dropout[0])
        # self.E = [tf.concat([self.user_embedding, self.item_embedding], 0)]
        # regularizer = tfv1.contrib.layers.l2_regularizer(self.reg)
        self.W1 = [tfv1.layers.Dense(self.dim, activation=tf.nn.leaky_relu, name=f'W1_{k}', use_bias=True) for k in range(self.conf.num_layers)]
        self.W2 = [tfv1.layers.Dense(self.dim, activation=tf.nn.leaky_relu, name=f'W2_{k}', use_bias=True) for k in range(self.conf.num_layers)]
        self.mess_dropout = [tfv1.layers.Dropout(self.conf.mess_dropout[k]) for k in range(self.conf.num_layers)]
        # self.W = tf.layers.Dense(self.dim*(1+self.conf.num_layers), activation=tf.nn.leaky_relu, name='W', use_bias=True)
        # self.Wh = tf.layers.Dense(self.dim*(1+self.conf.num_layers), name='Wh', use_bias=True)
        # self.Wf = tf.layers.Dense(self.dim*(1+self.conf.num_layers), name='Wf', use_bias=True)

    def constructTrainGraph(self):
        ego_embeddings_u = self.user_embedding
        ego_embeddings_i = self.item_embedding
        E_u = [self.user_embedding]
        E_i = [self.item_embedding]
        for k in range(self.conf.num_layers):
            consumed_items_sparse_matrix = self.construct_consumed_items_sparse_matrix(self.node_dropout(self.consumed_items_values_input, training=self.training))
            inter_users_sparse_matrix = self.construct_interacted_users_sparse_matrix(self.node_dropout(self.inter_users_values_input, training=self.training))
            tmp_embeddings_u = tf.sparse.sparse_dense_matmul(consumed_items_sparse_matrix, ego_embeddings_i)
            side_embeddings_u = tmp_embeddings_u + self.user_self_value*ego_embeddings_u
            tmp_embeddings_i = tf.sparse.sparse_dense_matmul(inter_users_sparse_matrix, ego_embeddings_u)
            side_embeddings_i = tmp_embeddings_i + self.item_self_value*ego_embeddings_i
            sum_embeddings_u = self.W1[k](side_embeddings_u)
            sum_embeddings_i = self.W1[k](side_embeddings_i)
            bi_embeddings_u = tf.multiply(ego_embeddings_u, tmp_embeddings_u)
            bi_embeddings_i = tf.multiply(ego_embeddings_i, tmp_embeddings_i)
            bi_embeddings_u = self.W2[k](bi_embeddings_u)
            bi_embeddings_i = self.W2[k](bi_embeddings_i)
            ego_embeddings_u = self.mess_dropout[k](sum_embeddings_u + bi_embeddings_u, training=self.training)
            ego_embeddings_i = self.mess_dropout[k](sum_embeddings_i + bi_embeddings_i, training=self.training)
            norm_embeddings_u = tf.math.l2_normalize(ego_embeddings_u, axis=1)
            norm_embeddings_i = tf.math.l2_normalize(ego_embeddings_i, axis=1)
            E_u.append(norm_embeddings_u)
            E_i.append(norm_embeddings_i)
        user_emb = tf.concat(E_u, 1)
        # self.user_emb = tf.nn.leaky_relu(self.Wh(self.user_emb) + self.Wf(self.W(self.user_side)))
        item_emb = tf.concat(E_i, 1)
        # self.user_emb, self.item_emb = tf.split(self.E_emb, [self.num_users, self.num_items], 0)

        self.prediction = self.predict(user_emb, item_emb)
        self.loss = self.BPRloss(user_emb, item_emb)
        # + tf.losses.get_regularization_loss()
        self.opt = tfv1.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        self.init = tfv1.global_variables_initializer()

    def saveVariables(self):
        variables_dict = {}
        variables_dict['user_embedding'] = self.user_embedding
        variables_dict['item_embedding'] = self.item_embedding
        # variables_dict['Wh/kernel'] = self.Wh.kernel
        # variables_dict['Wh/bias'] = self.Wh.bias
        # variables_dict['W/kernel'] = self.W.kernel
        # variables_dict['W/bias'] = self.W.bias
        # variables_dict['Wf/kernel'] = self.Wf.kernel
        # variables_dict['Wf/bias'] = self.Wf.bias
        for k in range(self.conf.num_layers):
            variables_dict['W1_{}/kernel'.format(k)] = self.W1[k].kernel
            variables_dict['W1_{}/bias'.format(k)] = self.W1[k].bias
            variables_dict['W2_{}/kernel'.format(k)] = self.W2[k].kernel
            variables_dict['W2_{}/bias'.format(k)] = self.W2[k].bias
        self.saver = tfv1.train.Saver(variables_dict)
