import tensorflow as tf
import numpy as np
import os
from base_model import base_model

class graphrec(base_model):
    def inputSupply(self, data_dict):
        self.social_edges_user0 = data_dict['social_edges_user0']
        self.social_edges_user1 = data_dict['social_edges_user1']
        self.interaction_user_indices = data_dict['interaction_user_indices']
        self.interaction_item_indices = data_dict['interaction_item_indices']

        # self.inter_users_values_input = data_dict['inter_users_values_input']
        self.social_neighbors_indices_input = np.concatenate([self.social_edges_user0, self.social_edges_user1], 1)
        # self.social_neighbors_values_input = data_dict['social_neighbors_values_input']
        self.consumed_items_indices_input = np.concatenate([self.interaction_user_indices, self.interaction_item_indices], 1)
        # self.consumed_items_values_input = data_dict['consumed_items_values_input']
        self.inter_users_indices_input = np.concatenate([self.interaction_item_indices, self.interaction_user_indices], 1)
        self.inter_users_dense_shape = np.array([self.conf.num_items, self.conf.num_users]).astype(np.int64)
        self.social_neighbors_dense_shape = np.array([self.conf.num_users, self.conf.num_users]).astype(np.int64)
        self.consumed_items_dense_shape = np.array([self.conf.num_users, self.conf.num_items]).astype(np.int64)

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

    def construct_inter_users_sparse_matrix(self, sp_value):
        return tf.SparseTensor(
            indices = self.inter_users_indices_input, 
            values = tf.squeeze(sp_value),
            dense_shape=self.inter_users_dense_shape
        )

    def initializeNodes(self):
        super(graphrec, self).initializeNodes()
        self.num_layers = self.conf.num_layers
        self.u_connect = tf.layers.Dense(\
                self.dim, activation=tf.nn.leaky_relu, name='u_connect', use_bias=True)
        self.W = tf.layers.Dense(\
                self.dim, activation=tf.nn.leaky_relu, name='W', use_bias=True)
        self.att_layer = [tf.layers.Dense(self.dim//(2**k), activation=tf.nn.leaky_relu, name='att_layer', use_bias=True) for k in range(self.num_layers)]
        self.att_out_layer = tf.layers.Dense(1, name='att_out_layer', use_bias=True)
        self.predict_layer = [tf.layers.Dense(self.dim//(2**k), activation=tf.nn.leaky_relu, name='predict_layer', use_bias=True) for k in range(self.num_layers)]
        self.predict_out_layer = tf.layers.Dense(1, name='predict_out_layer', use_bias=True)
        # self.bn = [tf.layers.BatchNormalization(momentum=0.5, name='bn') for _ in range(5)]
        
    def constructTrainGraph(self):
        u_inter_emb = tf.gather_nd(self.user_embedding, self.interaction_user_indices)
        i_inter_emb = tf.gather_nd(self.item_embedding, self.interaction_item_indices)
        ui_att = self.Att(u_inter_emb, i_inter_emb)
        ui_att_matrix = tf.sparse.softmax(self.construct_consumed_items_sparse_matrix(ui_att))
        ui_agg_emb = self.W(tf.sparse_tensor_dense_matmul(ui_att_matrix, self.item_embedding)) # h^I

        u0_social_emb = tf.gather_nd(self.user_embedding, self.social_edges_user0)
        u1_social_emb = tf.gather_nd(ui_agg_emb, self.social_edges_user1)
        uu_att = self.Att(u0_social_emb, u1_social_emb)
        uu_att_matrix = tf.sparse.softmax(self.construct_social_neighbors_sparse_matrix(uu_att))
        uu_agg_emb = self.W(tf.sparse_tensor_dense_matmul(uu_att_matrix, ui_agg_emb)) # h^S
        uu_next_emb = self.u_connect(tf.concat([ui_agg_emb, uu_agg_emb], 1))

        iu_att = self.Att(i_inter_emb, u_inter_emb)
        iu_att_matrix = tf.sparse.softmax(self.construct_inter_users_sparse_matrix(iu_att))
        iu_next_emb = self.W(tf.sparse_tensor_dense_matmul(iu_att_matrix, self.user_embedding))

        latest_user_embedding_latent = tf.gather_nd(uu_next_emb, self.user_input)
        latest_item_latent = tf.gather_nd(iu_next_emb, self.item_input)
        latest_item_neg_latent = tf.gather_nd(iu_next_emb, self.item_neg_input)
        # interaction_BPR_vector = tf.multiply(latest_user_embedding_latent, latest_item_latent - latest_item_neg_latent)
        score_p = self.compute_score(tf.concat([latest_user_embedding_latent, latest_item_latent], 1))
        score_n = self.compute_score(tf.concat([latest_user_embedding_latent, latest_item_neg_latent], 1))
        self.BPR_loss = -tf.reduce_sum(tf.log(tf.sigmoid(score_p-score_n)))

        # self.prediction = tf.sigmoid(tf.matmul(latest_user_embedding_latent, tf.transpose(iu_next_emb)))
        self.prediction = tf.sigmoid(score_p)
        self.reg_loss = tf.add_n([tf.reduce_sum(tf.square(latest_user_embedding_latent)), tf.reduce_sum(tf.square(latest_item_latent)), tf.reduce_sum(tf.square(latest_item_neg_latent))])
        self.loss = self.reg*self.reg_loss + self.BPR_loss
        self.opt = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        self.init = tf.global_variables_initializer()

    def compute_score(self, x):
    #     iu_next_emb = tf.gather_nd(iu_next_emb, i_input)
        # uu_next_emb = self.uu_layer[1](self.dp[0](tf.nn.leaky_relu(self.bn[0](self.uu_layer[0](uu_next_emb)))))
        # iu_next_emb = self.iu_layer[1](self.dp[1](tf.nn.leaky_relu(self.bn[1](self.iu_layer[0](iu_next_emb)))))
        for k in range(self.num_layers):
            x = self.predict_layer[k](x)
        x = self.predict_out_layer(x)
        return x

    def saveVariables(self):
        variables_dict = {}
        variables_dict['user_embedding'] = self.user_embedding
        variables_dict['item_embedding'] = self.item_embedding
        variables_dict['u_connect/kernel'] = self.u_connect.kernel
        variables_dict['u_connect/bias'] = self.u_connect.bias
        variables_dict['W/kernel'] = self.W.kernel
        variables_dict['W/bias'] = self.W.bias
        for k in range(self.num_layers):
            variables_dict['att_layer_{}/kernel'.format(k)] = self.att_layer[k].kernel
            variables_dict['att_layer_{}/bias'.format(k)] = self.att_layer[k].bias
            variables_dict['predict_layer_{}/kernel'.format(k)] = self.predict_layer[k].kernel
            variables_dict['predict_layer_{}/bias'.format(k)] = self.predict_layer[k].bias
        variables_dict['att_out_layer/kernel'] = self.att_out_layer.kernel
        variables_dict['att_out_layer/bias'] = self.att_out_layer.bias
        variables_dict['predict_out_layer/kernel'] = self.predict_out_layer.kernel
        variables_dict['predict_out_layer/bias'] = self.predict_out_layer.bias
        self.saver = tf.train.Saver(variables_dict)
        # for k in range(5):
        #     if k == 0:
        #         variables_dict_other['bn/gamma'] = self.bn[0].gamma
        #         variables_dict_other['bn/beta'] = self.bn[0].beta
        #     else:
        #         variables_dict_other['bn_{}/gamma'.format(k)] = self.bn[k].gamma
        #         variables_dict_other['bn_{}/beta'.format(k)] = self.bn[k].beta
        # self.saver_other = tf.train.Saver(variables_dict_other)
        
    def Att(self, self_emb, f_emb):
        att = tf.concat([self_emb, f_emb], 1)
        for k in range(self.num_layers):
            att = (self.att_layer[k](att))
        att = self.att_out_layer(att)
        return att