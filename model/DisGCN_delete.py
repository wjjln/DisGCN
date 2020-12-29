import tensorflow as tf
import numpy as np
import os
from base_model import base_model

class disgcn(base_model):
    def inputSupply(self, data_dict):
        first_item, second_item, first_friend, second_friend = data_dict['dis']
        self.first_user_friend, self.second_user_friend = first_friend['first_user_friend'], second_friend['second_user_friend']
        self.first_user_item, self.second_user_item = first_item['first_user_item'], second_item['second_user_item']

        self.social_neighbors_indices_input = np.concatenate([first_user_friend, second_user_friend], 0)
        self.consumed_items_indices_input = np.concatenate([self.first_user_item, self.second_user_item], 0)
        self.inter_users_indices_input = np.fliplr(self.consumed_items_indices_input)
        self.consumed_items_dense_shape = np.array([self.num_users, self.num_items]).astype(np.int64)
        self.social_neighbors_dense_shape = np.array([self.num_users, self.num_users]).astype(np.int64)
        self.inter_users_dense_shape = np.array([self.num_items, self.num_users]).astype(np.int64)

        self.first_item_idx, self.first_item_f = first_item['first_item_idx'], first_item['first_item_f']
        self.second_item_idx, self.second_item_f = second_item['second_item_idx'], second_item['second_item_f']
        self.first_friend_idx, self.first_friend_i = first_friend['first_friend_idx'], first_friend['first_friend_i']
        self.second_friend_idx, self.second_friend_i = second_friend['second_friend_idx'], second_friend['second_friend_i']


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
        super(disgcn, self).initializeNodes()
        self.cor_user = tf.placeholder("int32", [None])
        self.cor_item = tf.placeholder("int32", [None])
        self.pos_mask = tf.placeholder("float32", [None, self.num_negatives])
        self.neg_mask = tf.placeholder("float32", [None, self.num_negatives])
        # self.interest_mask = tf.placeholder("int32", [None, self.num_negatives])
        self.user_social_embedding = tf.Variable(
                tf.random_normal([self.num_users, self.conf.dimension], stddev=0.01), name='user_social_embedding')
        self.item_social_embedding = tf.Variable(
                tf.random_normal([self.num_items, self.conf.dimension], stddev=0.01), name='item_social_embedding')
        regularizer = tf.contrib.layers.l2_regularizer(self.reg)
        self.W_ui_social = [tf.layers.Dense(self.dim, activation=tf.nn.leaky_relu, name='W_ui_social', use_bias=True, \
                    kernel_regularizer=regularizer, bias_regularizer=regularizer) for _ in range(self.conf.num_layers)]
        self.W_uu_social = [tf.layers.Dense(self.dim, activation=tf.nn.leaky_relu, name='W_uu_social', use_bias=True, \
                    kernel_regularizer=regularizer, bias_regularizer=regularizer) for _ in range(self.conf.num_layers)]
        self.W_ui_interest = [tf.layers.Dense(self.dim, activation=tf.nn.leaky_relu, name='W_ui_interest', use_bias=True, \
                    kernel_regularizer=regularizer, bias_regularizer=regularizer) for _ in range(self.conf.num_layers)]
        self.W_uu_interest = [tf.layers.Dense(self.dim, activation=tf.nn.leaky_relu, name='W_uu_interest', use_bias=True, \
                    kernel_regularizer=regularizer, bias_regularizer=regularizer) for _ in range(self.conf.num_layers)]
    
    def Gen_att(self, emb, pair, idx):
        h = tf.gather(emb[0], pair[:, 0])
        t = tf.gather(emb[1], pair[:, 1])
        r = tf.gather(emb[2], pair[:, 2])
        att = tf.nn.sigmoid(tf.segment_sum(1.0/(1.0+tf.reduce_sum(\
                tf.abs((h-t)*r), -1)), idx))
        return att

    def neighbor_rooting(self, emb, pair):
        h_s, h_i = tf.gather(emb[0], pair[:, 0]), tf.gather(emb[1], pair[:, 0])
        t_s, t_i = tf.gather(emb[2], pair[:, 1]), tf.gather(emb[3], pair[:, 1])
        s = tf.reduce_sum(h_s*t_s, -1)
        i = tf.reduce_sum(h_i*t_i, -1)
        s_add_i = tf.exp(s)+tf.exp(i)
        s, i = s/s_add_i, i/s_add_i
        return s, i

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

        def _create_distance_covariance(D1, D2):
            # calculate distance covariance between D1 and D2
            n_samples = tf.dtypes.cast(tf.shape(D1)[0], tf.float32)
            dcov = tf.sqrt(tf.maximum(tf.reduce_sum(D1 * D2) / (n_samples * n_samples), 0.0) + 1e-8)
            # dcov = tf.sqrt(tf.maximum(tf.reduce_sum(D1 * D2)) / n_samples
            return dcov

        D1 = _create_centered_distance(X1)
        D2 = _create_centered_distance(X2)

        dcov_12 = _create_distance_covariance(D1, D2)
        dcov_11 = _create_distance_covariance(D1, D1)
        dcov_22 = _create_distance_covariance(D2, D2)

        # calculate the distance correlation
        dcor = dcov_12 / (tf.sqrt(tf.maximum(dcov_11 * dcov_22, 0.0)) + 1e-10)
        # return tf.reduce_sum(D1) + tf.reduce_sum(D2)
        return dcor
        
    def constructTrainGraph(self):
        tmp_user_social_embedding, tmp_user_embedding, tmp_item_social_embedding, tmp_item_embedding = \
                    self.user_social_embedding, self.user_embedding, self.item_social_embedding, self.item_embedding
        user_social_embedding_list, user_embedding_list, item_social_embedding_list, item_embedding_list = \
                    [self.user_social_embedding], [self.user_embedding], [self.item_social_embedding], [self.item_embedding]
        for k in range(self.conf.num_layers):
            #----------------------u-i----------------------
            first_social_att = self.Gen_att([tmp_user_social_embedding, tmp_item_social_embedding, \
                    tmp_user_social_embedding], self.first_item_f, self.first_item_idx)
            first_interest_att = tf.exp(-first_social_att)
            second_social_att = self.Gen_att([tmp_user_social_embedding, tmp_item_social_embedding, \
                    tmp_user_social_embedding], self.second_item_f, self.second_item_idx)
            second_interest_att = tf.exp(-second_social_att)-tf.exp(-1.0)
            def_social_att = tf.concat([first_social_att, second_social_att], 0)
            def_interest_att = tf.concat([first_interest_att, second_interest_att], 0)

            emb_ui = [tmp_user_social_embedding, tmp_user_embedding, tmp_item_social_embedding, tmp_item_embedding]
            for t in range(self.conf.T):
                first_s, first_i = self.neighbor_rooting(emb_ui, self.first_user_item)
                second_s, second_i = self.neighbor_rooting(emb_ui, self.second_user_item)
                dis_social_att = tf.concat([first_s, second_s], 0)
                dis_interest_att = tf.concat([first_i, second_i], 0)
                if t < self.conf.T-1:
                    iu_social_matrix = self.construct_consumed_items_sparse_matrix(dis_social_att)
                    iu_interest_matrix = self.construct_consumed_items_sparse_matrix(dis_interest_att)
                    emb_ui[0] = tf.math.l2_normalize(tf.sparse_tensor_dense_matmul(iu_social_matrix, tmp_item_social_embedding), axis=1)
                    emb_ui[1] = tf.math.l2_normalize(tf.sparse_tensor_dense_matmul(iu_interest_matrix, tmp_item_embedding), axis=1)

            social_att = def_social_att + dis_social_att
            interest_att = def_interest_att + dis_interest_att
            iu_social_matrix = self.construct_consumed_items_sparse_matrix(social_att)
            iu_interest_matrix = self.construct_consumed_items_sparse_matrix(interest_att)
            ui_social_matrix = self.construct_interacted_users_sparse_matrix(social_att)
            ui_interest_matrix = self.construct_interacted_users_sparse_matrix(interest_att)
            

            #----------------------u-u----------------------
            first_interest_att = self.Gen_att([tmp_user_embedding, tmp_user_embedding, tmp_item_embedding], \
                    self.first_friend_i, self.first_friend_idx)
            first_social_att = tf.exp(-first_interest_att)
            second_interest_att = self.Gen_att([tmp_user_embedding, tmp_user_embedding, tmp_item_embedding], \
                    self.second_friend_i, self.second_friend_idx)
            second_social_att = tf.exp(-second_interest_att)-tf.exp(-1.0)
            def_social_att = tf.concat([first_social_att, second_social_att], 0)
            def_interest_att = tf.concat([first_interest_att, second_interest_att], 0)

            emb_uu = [tmp_user_social_embedding, tmp_user_embedding, tmp_user_social_embedding, tmp_user_embedding]
            for t in range(self.conf.T):
                first_s, first_i = self.neighbor_rooting(emb_uu, self.first_user_friend)
                second_s, second_i = self.neighbor_rooting(emb_uu, self.second_user_friend)
                dis_social_att = tf.concat([first_s, second_s], 0)
                dis_interest_att = tf.concat([first_i, second_i], 0)
                if t < self.conf.T-1:
                    uu_social_matrix = self.construct_social_neighbors_sparse_matrix(dis_social_att)
                    uu_interest_matrix = self.construct_social_neighbors_sparse_matrix(dis_interest_att)
                    emb_uu[0] = emb_uu[2] = tf.math.l2_normalize(tf.sparse_tensor_dense_matmul(uu_social_matrix, tmp_user_social_embedding), axis=1)
                    emb_uu[1] = emb_uu[3] = tf.math.l2_normalize(tf.sparse_tensor_dense_matmul(uu_interest_matrix, tmp_user_embedding), axis=1)

            social_att = def_social_att + dis_social_att
            interest_att = def_interest_att + dis_interest_att
            uu_social_matrix = self.construct_social_neighbors_sparse_matrix(social_att)
            uu_interest_matrix = self.construct_social_neighbors_sparse_matrix(interest_att)


            #----------------------propagation----------------------
            tmp_user_social_embedding = self.W_ui_social[k](tf.sparse_tensor_dense_matmul(iu_social_matrix, tmp_item_social_embedding))\
                    + self.W_uu_social[k](tf.sparse_tensor_dense_matmul(uu_social_matrix, tmp_user_social_embedding))
            tmp_user_embedding = self.W_ui_interest[k](tf.sparse_tensor_dense_matmul(iu_interest_matrix, tmp_item_embedding))\
                    + self.W_uu_interest[k](tf.sparse_tensor_dense_matmul(uu_interest_matrix, tmp_user_embedding))
            tmp_item_social_embedding = self.W_ui_social[k](tf.sparse_tensor_dense_matmul(ui_social_matrix, tmp_user_social_embedding))
            tmp_item_embedding = self.W_ui_interest[k](tf.sparse_tensor_dense_matmul(ui_interest_matrix, tmp_user_embedding))
            user_social_embedding_list.append(tmp_user_social_embedding)
            user_embedding_list.append(tmp_social_embedding)
            item_social_embedding_list.append(tmp_item_social_embedding)
            item_embedding_list.append(tmp_item_embedding)
        
        user_social_embedding, user_interest_embedding = tf.concat(user_social_embedding_list, 1), tf.concat(user_embedding_list, 1)
        item_social_embedding, item_interest_embedding = tf.concat(item_social_embedding_list, 1), tf.concat(item_embedding_list, 1)
        user_embedding = tf.concat([user_social_embedding, user_interest_embedding], 1)
        item_embedding = tf.concat([item_social_embedding, item_interest_embedding], 1)
        self.predict(user_embedding, item_embedding)
        self.click_loss = self.BPRloss(user_embedding, item_embedding)
        self.social_loss = self.BPRloss(user_social_embedding, item_social_embedding, self.pos_mask) \
                + self.BPRloss(user_social_embedding, item_social_embedding, self.neg_mask, -1)
        self.interest_loss = self.BPRloss(user_interest_embedding, item_interest_embedding, self.neg_mask)
        cor_user_social_embedding, cor_user_interest_embedding = \
                tf.gather(user_social_embedding, self.cor_user), tf.gather(user_interest_embedding, self.cor_user)
        cor_item_social_embedding, cor_item_interest_embedding = \
                tf.gather(item_social_embedding, self.cor_item), tf.gather(item_interest_embedding, self.cor_item)
        self.cor_loss = self._create_distance_correlation(cor_user_social_embedding, cor_user_interest_embedding) \
                + self._create_distance_correlation(cor_item_social_embedding, cor_item_interest_embedding)
        self.loss = self.click_loss + self.conf.alpha*(self.social_loss+self.interest_loss) + self.conf.beta*self.cor_loss
        self.Adam = tf.train.AdamOptimizer(self.learning_rate)
        self.opt = self.Adam.minimize(self.loss)
        self.init = tf.global_variables_initializer()

    def saveVariables(self):
        variables_dict = {}
        variables_dict['user_embedding'] = self.user_embedding
        variables_dict['item_embedding'] = self.item_embedding
        variables_dict['user_social_embedding'] = self.user_social_embedding
        variables_dict['item_social_embedding'] = self.item_social_embedding
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
        self.saver = tf.train.Saver(variables_dict)

    def defineMap(self):
        super(disgcn, self).defineMap()
        tmp_mask = {self.pos_mask:'pos_mask', self.neg_mask:'neg_mask', self.cor_user:'cor_user', self.cor_item:'cor_item'}
        tmp_out = [self.loss, self.click_loss, self.social_loss, self.interest_loss, self.cor_loss]
        self.map_dict['train'].update(tmp_mask)
        self.map_dict['out']['train'] = tmp_out
        for k in ('test', 'val'):
            self.map_dict['out'][k] = self.click_loss