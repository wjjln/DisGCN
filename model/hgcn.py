import tensorflow as tf
import numpy as np
import os
from base_model import base_model

class hgcn(base_model):
    def inputSupply(self, data_dict):
        self.dp = tf.placeholder("float32", [1])
        self.dp_layer = tf.layers.Dropout(tf.gather(self.dp, 0))
        self.interaction_user_indices = data_dict['interaction_user_indices']
        self.interaction_item_indices = data_dict['interaction_item_indices']
        self.user_self_value = data_dict['user_self_value']
        self.item_self_value = data_dict['item_self_value']
        self.consumed_items_indices_input = np.concatenate([self.interaction_user_indices, self.interaction_item_indices], 1)
        self.consumed_items_values_input = data_dict['consumed_items_values_input_GCN']
        self.consumed_items_values_input = self.dp_layer(self.consumed_items_values_input)
        self.inter_users_indices_input = np.concatenate([self.interaction_item_indices, self.interaction_user_indices], 1)
        self.inter_users_values_input = data_dict['inter_users_values_input_GCN']
        self.inter_users_values_input = self.dp_layer(self.inter_users_values_input)
        self.consumed_items_dense_shape = np.array([self.conf.num_users, self.conf.num_items]).astype(np.int64)
        self.inter_users_dense_shape = np.array([self.conf.num_items, self.conf.num_users]).astype(np.int64)

        self.consumed_items_sparse_matrix = self.construct_consumed_items_sparse_matrix(self.consumed_items_values_input)
        self.inter_users_sparse_matrix = self.construct_inter_users_sparse_matrix(self.inter_users_values_input)

    def initializeNodes(self):
        super(hgcn, self).initializeNodes()
        self.num_layers = self.conf.num_layers
        self.W1 = [tf.layers.Dense(self.conf.dimension, activation=tf.nn.leaky_relu, name='W1', use_bias=True) for _ in range(self.num_layers)]
        self.relation_mean = tf.Variable(tf.random_normal([self.conf.num_mem, self.dim], mean=0.0, stddev=0.01), name='relation_mean')
        self.relation_var = tf.Variable(tf.random_normal([self.conf.num_mem, self.dim], stddev=0.01), name='relation_var')
        self.m = tf.distributions.Normal(0.0, 1.0)
        self.W_z = [tf.layers.Dense(100, activation=tf.nn.leaky_relu, name='W_z', use_bias=True), tf.layers.Dense(self.conf.num_mem, activation=tf.nn.leaky_relu, name='W_z', use_bias=True)]
        self.s_embL2 = 0
        # self.alpha = tf.Variable(tf.random_normal([1], mean=0.0, stddev=1), name='alpha')
        self.Wr = tf.layers.Dense(self.dim, activation=tf.nn.leaky_relu, name='Wr', use_bias=True)
        # self.K = tf.Variable(tf.random_normal([self.conf.num_mem, self.dim], stddev=0.01), name='K')
        # self.M = tf.Variable(tf.random_normal([self.conf.num_mem, self.dim], stddev=0.01), name='M')
        # self.W_K = tf.layers.Dense(self.dim, activation=tf.nn.leaky_relu, name='W_K', use_bias=True)

    def edge_gen(self, user_emb, item_emb, user, item):
        u_emb = tf.gather_nd(user_emb, user)
        i_emb = tf.gather_nd(item_emb, item)
        edges_preference_cat = self.W_z[1](self.W_z[0](tf.concat([u_emb, i_emb, u_emb*i_emb], 1)))
        edges_preference_cat_softmax = tf.nn.softmax(edges_preference_cat)
        m_sample = self.m.sample([self.conf.num_mem, self.dim])
        M = self.relation_mean + self.relation_var*m_sample
        edges_preference = tf.matmul(edges_preference_cat_softmax, M)
        loss = -tf.reduce_sum(tf.square(self.Wr(u_emb)+edges_preference-self.Wr(i_emb)), 1)
        self.s_embL2 += tf.add_n([tf.reduce_sum(tf.square(u_emb)), tf.reduce_sum(tf.square(i_emb))])
        # edges_preference = tf.matmul(tf.nn.softmax(tf.matmul(self.W_K(tf.concat([u_emb, i_emb, u_emb*i_emb], 1)), tf.transpose(self.K))), self.M)
        return loss

    def propagate(self, user_p, layer, item_e=None):
        user_embedding_from_item = tf.sparse_tensor_dense_matmul(self.consumed_items_sparse_matrix, item_e)
        item_embedding_from_user = tf.sparse_tensor_dense_matmul(self.inter_users_sparse_matrix, user_p)
        user_embedding_last = self.W1[layer](user_embedding_from_item + self.user_self_value*user_p)
        item_embedding_last = self.W1[layer](item_embedding_from_user + self.item_self_value*item_e)
        return user_embedding_last, item_embedding_last 

    def constructTrainGraph(self):
        tmp_user_embedding, tmp_item_embedding = self.user_embedding, self.item_embedding
        user_embedding_last, item_embedding_last = [self.user_embedding], [self.item_embedding]
        self.reg_p = self.edge_gen(tmp_user_embedding, tmp_item_embedding, self.user_input, self.item_input)
        self.reg_n = self.edge_gen(tmp_user_embedding, tmp_item_embedding, self.user_input, self.item_neg_input)
        for k in range(self.num_layers):
            tmp_user_embedding, tmp_item_embedding = self.propagate(tmp_user_embedding, k, tmp_item_embedding)
            user_embedding_last.append(tf.math.l2_normalize(tmp_user_embedding, axis=1))
            item_embedding_last.append(tf.math.l2_normalize(tmp_item_embedding, axis=1))
        user_embedding_last = tf.concat(user_embedding_last, 1)
        item_embedding_last = tf.concat(item_embedding_last, 1)
        latest_user_embedding_latent = tf.gather_nd(user_embedding_last, self.user_input)
        latest_item_latent = tf.gather_nd(item_embedding_last, self.item_input)
        latest_item_neg_latent = tf.gather_nd(item_embedding_last, self.item_neg_input)
        W_loss, s_W_loss = 0, 0
        for k in range(self.num_layers):
            W_loss += (tf.reduce_sum(tf.square(self.W1[k].kernel)) + tf.reduce_sum(tf.square(self.W1[k].bias)))
        # for k in range(self.num_layers+1):
        s_W_loss += (tf.reduce_sum(tf.square(self.W_z[0].kernel)) + tf.reduce_sum(tf.square(self.W_z[0].bias))\
                        + tf.reduce_sum(tf.square(self.W_z[1].kernel)) + tf.reduce_sum(tf.square(self.W_z[1].bias)))
        s_W_loss += tf.reduce_sum(tf.square(self.relation_mean))
        s_W_loss += tf.reduce_sum(tf.square(self.relation_var))
        s_W_loss += (tf.reduce_sum(tf.square(self.Wr.kernel)) + tf.reduce_sum(tf.square(self.Wr.bias)))
        # W_loss += tf.reduce_sum(tf.square(self.K))
        # W_loss += tf.reduce_sum(tf.square(self.M))
        # W_loss += (tf.reduce_sum(tf.square(self.W_K.kernel)) + tf.reduce_sum(tf.square(self.W_K.bias)))
        interaction_BPR_vector = tf.multiply(latest_user_embedding_latent, latest_item_latent - latest_item_neg_latent)
        self.prediction = tf.sigmoid(tf.matmul(latest_user_embedding_latent, tf.transpose(item_embedding_last)))
        self.reg_loss = tf.add_n([tf.reduce_sum(tf.square(latest_user_embedding_latent)), tf.reduce_sum(tf.square(latest_item_latent)), tf.reduce_sum(tf.square(latest_item_neg_latent)), W_loss])
        self.s_reg_loss = tf.add_n([self.s_embL2, s_W_loss])
        self.BPR_loss = tf.reduce_sum(tf.log(tf.sigmoid(tf.reduce_sum(interaction_BPR_vector, 1, keepdims=True))))
        self.loss = self.reg * self.reg_loss - self.BPR_loss
        self.opt = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        self.simloss = self.reg*self.s_reg_loss - tf.reduce_sum(tf.log(tf.sigmoid(self.reg_p - self.reg_n)))
        self.sim_opt = tf.train.AdamOptimizer(self.learning_rate).minimize(self.simloss)
        self.init = tf.global_variables_initializer()

    def saveVariables(self):
        variables_dict = {}
        variables_dict['user_embedding'] = self.user_embedding
        variables_dict['item_embedding'] = self.item_embedding
        variables_dict['relation_mean'] = self.relation_mean
        variables_dict['relation_var'] = self.relation_var
        variables_dict['W_z/kernel'] = self.W_z[0].kernel
        variables_dict['W_z/bias'] = self.W_z[0].bias
        variables_dict['W_z_1/kernel'] = self.W_z[1].kernel
        variables_dict['W_z_1/bias'] = self.W_z[1].bias
        variables_dict['Wr/kernel'] = self.Wr.kernel
        variables_dict['Wr/bias'] = self.Wr.bias
        # variables_dict['K'] = self.K
        # variables_dict['M'] = self.M
        for k in range(self.conf.num_layers):
            variables_dict['W1_{}/kernel'.format(k)] = self.W1[k].kernel
            variables_dict['W1_{}/bias'.format(k)] = self.W1[k].bias
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