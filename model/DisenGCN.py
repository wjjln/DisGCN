import tensorflow as tf
import tensorflow.compat.v1 as tfv1
import numpy as np
import os
from model.base_model import base_model

class disengcn(base_model):

    def __init__(self, conf, reg, learning_rate):
        super(disengcn, self).__init__(conf, reg, learning_rate)
        self.beta = conf.beta
        self.w = conf.w

    def inputSupply(self, data_dict):
        super(disengcn, self).inputSupply(data_dict)
        self.interaction_user_indices = data_dict['interaction_user_indices']
        self.interaction_item_indices = data_dict['interaction_item_indices']
        self.social_edges_user0 = data_dict['social_edges_user0']
        self.social_edges_user1 = data_dict['social_edges_user1']

        self.consumed_items_indices_input = np.concatenate([self.interaction_user_indices, self.interaction_item_indices], 1)
        self.consumed_items_values_input = data_dict['consumed_items_values_input']
        self.consumed_items_dense_shape = np.array([self.conf.num_users, self.conf.num_items]).astype(np.int64)
        # self.iu_social_matrix = self.construct_consumed_items_sparse_matrix(self.consumed_items_values_input)
        # self.iu_interest_matrix = self.construct_consumed_items_sparse_matrix(self.consumed_items_values_input)

        self.inter_users_indices_input = np.concatenate([self.interaction_item_indices, self.interaction_user_indices], 1)
        self.inter_users_values_input = data_dict['inter_users_values_input']
        self.inter_users_dense_shape = np.array([self.conf.num_items, self.conf.num_users]).astype(np.int64)
        # self.ui_social_matrix = self.construct_inter_users_sparse_matrix(self.inter_users_values_input)
        # self.ui_interest_matrix = self.construct_inter_users_sparse_matrix(self.inter_users_values_input)

        self.social_neighbors_indices_input = np.concatenate([self.social_edges_user0, self.social_edges_user1], 1)
        self.social_neighbors_values_input = data_dict['social_neighbors_values_input']
        self.social_neighbors_dense_shape = np.array([self.num_users, self.num_users]).astype(np.int64)
        # self.uu_social_matrix = self.construct_social_neighbors_sparse_matrix(self.social_neighbors_values_input)
        # self.uu_interest_matrix = self.construct_social_neighbors_sparse_matrix(self.social_neighbors_values_input)
        

    def construct_social_neighbors_sparse_matrix(self, sp_value):
        return tf.SparseTensor(
            indices = self.social_neighbors_indices_input,
            values = tf.squeeze(sp_value),
            dense_shape=self.social_neighbors_dense_shape
        )
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
        super(disengcn, self).initializeNodes()
        self.cor_idx = tfv1.placeholder("int32", [None])
        self.user_social_embedding = tf.Variable(
                tf.random.normal([self.num_users, self.conf.dimension], stddev=0.01), name='user_social_embedding')
        self.item_social_embedding = tf.Variable(
                tf.random.normal([self.num_items, self.conf.dimension], stddev=0.01), name='item_social_embedding')
        if self.w:
            # regularizer = tf.contrib.layers.l2_regularizer(self.reg)
            # kernel_regularizer=regularizer, bias_regularizer=regularizer
            self.W_ui_social = [tfv1.layers.Dense(self.dim, activation=tf.nn.leaky_relu, name='W_ui_social', use_bias=True \
                        ) for _ in range(self.conf.num_layers)]
            self.W_uu_social = [tfv1.layers.Dense(self.dim, activation=tf.nn.leaky_relu, name='W_uu_social', use_bias=True \
                        ) for _ in range(self.conf.num_layers)]
            self.W_ui_interest = [tfv1.layers.Dense(self.dim, activation=tf.nn.leaky_relu, name='W_ui_interest', use_bias=True \
                        ) for _ in range(self.conf.num_layers)]
            self.W_uu_interest = [tfv1.layers.Dense(self.dim, activation=tf.nn.leaky_relu, name='W_uu_interest', use_bias=True \
                        ) for _ in range(self.conf.num_layers)]
    

    def _create_distance_correlation(self, X1, X2):

        def _create_centered_distance(X):
            
            # calculate the pairwise distance of X
            # .... A with the size of [batch_size, embed_size/n_factors]
            # .... D with the size of [batch_size, batch_size]
            # X = tf.math.l2_normalize(XX, axis=1)

            r = tf.reduce_sum(tf.square(X), 1, keepdims=True)
            D = tf.sqrt(tf.maximum(r - 2 * tf.matmul(a=X, b=X, transpose_b=True) + tf.transpose(r), 0.0) + 1e-8)

            # # calculate the centered distance of X
            # # .... D with the size of [batch_size, batch_size]
            D = D - tf.reduce_mean(D, axis=0, keepdims=True) - tf.reduce_mean(D, axis=1, keepdims=True) \
                + tf.reduce_mean(D)
            return D

        n_samples = tf.dtypes.cast(tf.shape(X1)[0], tf.float32)

        def _create_distance_covariance(D1, D2):
            # calculate distance covariance between D1 and D2
            dcov = tf.sqrt(tf.maximum(tf.reduce_sum(D1 * D2) / (n_samples**2), 0.0) + 1e-8)
            # dcov = tf.sqrt(tf.maximum(tf.reduce_sum(D1 * D2)) / n_samples
            return dcov

        D1 = _create_centered_distance(X1)
        D2 = _create_centered_distance(X2)

        dcov_12 = _create_distance_covariance(D1, D2)
        dcov_11 = _create_distance_covariance(D1, D1)
        dcov_22 = _create_distance_covariance(D2, D2)

        # calculate the distance correlation
        dcor = dcov_12 / (tf.sqrt(tf.maximum(dcov_11 * dcov_22, 0.0)) + 1e-10) * n_samples/2
        # return tf.reduce_sum(D1) + tf.reduce_sum(D2)
        return dcor
        
    def neighbor_rooting(self, emb, pair):
        h_s, h_i = tf.gather(emb[0], pair[:, 0]), tf.gather(emb[1], pair[:, 0])
        t_s, t_i = tf.gather(emb[2], pair[:, 1]), tf.gather(emb[3], pair[:, 1])
        s = tf.reduce_sum(h_s*t_s, -1)
        i = tf.reduce_sum(h_i*t_i, -1)
        s_add_i = tf.exp(s)+tf.exp(i)
        s, i = s/s_add_i, i/s_add_i
        return s, i

    def constructTrainGraph(self):
        user_social_embedding, user_embedding, item_social_embedding, item_embedding = \
                    self.user_social_embedding, self.user_embedding, self.item_social_embedding, self.item_embedding
        tmp_user_social_embedding, tmp_user_embedding, tmp_item_social_embedding, tmp_item_embedding = \
                    self.user_social_embedding, self.user_embedding, self.item_social_embedding, self.item_embedding
        user_social_embedding_list, user_embedding_list, item_social_embedding_list, item_embedding_list = \
                    [self.user_social_embedding], [self.user_embedding], [self.item_social_embedding], [self.item_embedding]
        for k in range(self.conf.num_layers):
            
            for t in range(self.conf.iter):
                emb_ui = [user_social_embedding, user_embedding, tmp_item_social_embedding, tmp_item_embedding]
                emb_iu = [item_social_embedding, item_embedding, tmp_user_social_embedding, tmp_user_embedding]
                emb_uu = [user_social_embedding, user_embedding, tmp_user_social_embedding, tmp_user_embedding]

                ui_s, ui_i = self.neighbor_rooting(emb_ui, self.consumed_items_indices_input)
                self.ui_social_matrix = self.construct_interacted_users_sparse_matrix(ui_s)
                self.ui_interest_matrix = self.construct_interacted_users_sparse_matrix(ui_i)
                iu_s, iu_i = self.neighbor_rooting(emb_iu, self.inter_users_indices_input)
                self.iu_social_matrix = self.construct_consumed_items_sparse_matrix(iu_s)
                self.iu_interest_matrix = self.construct_consumed_items_sparse_matrix(iu_i)
                uu_s, uu_i = self.neighbor_rooting(emb_uu, self.social_neighbors_indices_input)
                self.uu_social_matrix = self.construct_social_neighbors_sparse_matrix(uu_s)
                self.uu_interest_matrix = self.construct_social_neighbors_sparse_matrix(uu_i)
                
                user_social_from_item = tf.sparse.sparse_dense_matmul(self.iu_social_matrix, item_social_embedding)
                user_social_from_friend = tf.sparse.sparse_dense_matmul(self.uu_social_matrix, user_social_embedding)
                user_interest_from_item = tf.sparse.sparse_dense_matmul(self.iu_interest_matrix, item_embedding)
                user_interest_from_friend = tf.sparse.sparse_dense_matmul(self.uu_interest_matrix, user_embedding)
                item_social_from_user = tf.sparse.sparse_dense_matmul(self.ui_social_matrix, user_social_embedding)
                item_interest_from_user = tf.sparse.sparse_dense_matmul(self.ui_interest_matrix, user_embedding)
                if self.w and t == self.conf.iter-1:
                    user_social_from_item, user_social_from_friend = self.W_ui_social[k](user_social_from_item), self.W_uu_social[k](user_social_from_friend)
                    user_interest_from_item, user_interest_from_friend = self.W_ui_interest[k](user_interest_from_item), self.W_uu_interest[k](user_interest_from_friend)
                    item_social_from_user, item_interest_from_user = self.W_ui_social[k](item_social_from_user), self.W_ui_interest[k](item_interest_from_user)
                tmp_user_social_embedding = tf.math.l2_normalize(user_social_from_friend + user_social_from_item, 1)
                tmp_user_embedding = tf.math.l2_normalize(user_interest_from_friend + user_interest_from_item, 1)
                tmp_item_social_embedding = tf.math.l2_normalize(item_social_from_user, 1)
                tmp_item_embedding = tf.math.l2_normalize(item_interest_from_user, 1)
            
            user_social_embedding, user_embedding, item_social_embedding, item_embedding = \
                        tmp_user_social_embedding, tmp_user_embedding, tmp_item_social_embedding, tmp_item_embedding
            user_social_embedding_list.append(tmp_user_social_embedding)
            user_embedding_list.append(tmp_user_embedding)
            item_social_embedding_list.append(tmp_item_social_embedding)
            item_embedding_list.append(tmp_item_embedding)
        
        user_social_embedding, user_interest_embedding = tf.concat(user_social_embedding_list, 1), tf.concat(user_embedding_list, 1)
        item_social_embedding, item_interest_embedding = tf.concat(item_social_embedding_list, 1), tf.concat(item_embedding_list, 1)
        # user_embedding = tf.concat([user_social_embedding, user_interest_embedding], 1)
        # item_embedding = tf.concat([item_social_embedding, item_interest_embedding], 1)
        # self.predict(user_embedding, item_embedding)
        # self.click_loss = self.BPRloss(user_embedding, item_embedding)
        social_user_from_friend = tf.sparse.sparse_dense_matmul(self.uu_social_matrix, user_social_embedding)
        social_item_from_user = tf.sparse.sparse_dense_matmul(self.ui_social_matrix, user_social_embedding)
        self.click_loss = self.Dis_BPRloss([user_interest_embedding, item_interest_embedding, user_social_embedding, item_social_embedding, \
                                            social_user_from_friend, social_item_from_user])
        
        cor_social_embedding = tf.gather(tf.concat([user_social_embedding, item_social_embedding], 0), self.cor_idx)
        cor_interest_embedding = tf.gather(tf.concat([user_interest_embedding, item_interest_embedding], 0), self.cor_idx)
        self.cor_loss = self._create_distance_correlation(cor_social_embedding, cor_interest_embedding)
        self.loss = self.click_loss + self.beta*self.cor_loss
        self.Adam = tfv1.train.AdamOptimizer(self.learning_rate)
        self.opt = self.Adam.minimize(self.loss)
        self.init = tfv1.global_variables_initializer()

    def saveVariables(self):
        variables_dict = {}
        variables_dict['user_embedding'] = self.user_embedding
        variables_dict['item_embedding'] = self.item_embedding
        variables_dict['user_social_embedding'] = self.user_social_embedding
        variables_dict['item_social_embedding'] = self.item_social_embedding
        if self.w:
            for k in range(self.conf.num_layers):
                variables_dict['W_ui_social_{}/kernel'.format(k)] = self.W_ui_social[k].kernel
                variables_dict['W_ui_social_{}/bias'.format(k)] = self.W_ui_social[k].bias
                variables_dict['W_ui_interest_{}/kernel'.format(k)] = self.W_ui_interest[k].kernel
                variables_dict['W_ui_interest_{}/bias'.format(k)] = self.W_ui_interest[k].bias
                variables_dict['W_uu_social_{}/kernel'.format(k)] = self.W_uu_social[k].kernel
                variables_dict['W_uu_social_{}/bias'.format(k)] = self.W_uu_social[k].bias
                variables_dict['W_uu_interest_{}/kernel'.format(k)] = self.W_uu_interest[k].kernel
                variables_dict['W_uu_interest_{}/bias'.format(k)] = self.W_uu_interest[k].bias

        variables_dict['Adam'] = self.Adam
        self.saver = tfv1.train.Saver(variables_dict)

    def defineMap(self):
        super(disengcn, self).defineMap()
        tmp_mask = {self.cor_idx:'cor_idx'}
        tmp_out = [self.loss, self.click_loss, self.cor_loss]
        self.map_dict['train'].update(tmp_mask)
        self.map_dict['out']['train'] = tmp_out
        for k in ('test', 'val'):
            self.map_dict['out'][k] = self.click_loss