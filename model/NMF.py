import tensorflow as tf
import numpy as np
import os
from base_model import base_model

class nmf(base_model):
    def __init__(self, conf, reg, learning_rate):
        # super(nmf, self).__init__(conf, reg, learning_rate)
        self.conf = conf
        self.model_type = 'nmf'
        self.pretrain_data = conf.pretrain_flag

        self.n_users = conf.num_users
        self.n_items = conf.num_items

        self.lr = learning_rate

        self.emb_dim = conf.dimension

        self.weight_size = [128, 64, 32]
        self.n_layers = len(self.weight_size)

        self.model_type += '_l%d' % self.n_layers

        self.regs = reg
        # self.decay = self.regs[-1]

        # self.verbose = args.verbose

        # placeholder definition

        self.dropout_keep = [0.1, 0.1, 0.1]
        # self.train_phase = tf.placeholder(tf.bool)

        # self.global_step = tf.Variable(0, trainable=False)
    def inputSupply(self, data_dict):
        pass

    def initializeNodes(self):
        self.item_input = tf.placeholder("int32", [None, 1])
        self.user_input = tf.placeholder("int32", [None, 1])
        self.item_neg_input = tf.placeholder("int32", [None, 1])
        self.unique_user_list = tf.placeholder("int32", [None, 1])
        self.unique_item_list = tf.placeholder("int32", [None, 1])
        self.unique_item_neg_list = tf.placeholder("int32", [None, 1])

        self.weights = self._init_weights()

        # Original embedding.
    def constructTrainGraph(self):
        u_e = tf.gather_nd(self.weights['user_embedding'], self.user_input)
        pos_i_e = tf.gather_nd(self.weights['item_embedding'], self.item_input)
        neg_i_e = tf.gather_nd(self.weights['item_embedding'], self.item_neg_input)

        # All ratings for all users.
        # self.batch_ratings = self._create_batch_ratings(u_e, pos_i_e)

        self.mf_loss, self.emb_loss, self.reg_loss = self.create_bpr_loss(u_e, pos_i_e, neg_i_e)
        self.loss = self.mf_loss + self.emb_loss + self.reg_loss

        # self.dy_lr = tf.train.exponential_decay(self.lr, self.global_step, 10000, self.lr_decay, staircase=True)
        # self.opt = tf.train.RMSPropOptimizer(learning_rate=self.dy_lr).minimize(self.loss, global_step=self.global_step)
        self.opt = tf.train.RMSPropOptimizer(learning_rate=self.lr).minimize(self.loss)
        self.init = tf.global_variables_initializer()
        # self.updates = self.opt.minimize(self.loss, var_list=self.weights)

        # self._statistics_params()

    def _init_weights(self):
        all_weights = dict()

        initializer = tf.contrib.layers.xavier_initializer()
        if self.pretrain_data == 0:
            all_weights['user_embedding'] = tf.Variable(initializer([self.n_users, self.emb_dim]), name='user_embedding')
            all_weights['item_embedding'] = tf.Variable(initializer([self.n_items, self.emb_dim]), name='item_embedding')
            print('using xavier initialization')
        else:
            pre_emb = np.load(os.path.join(os.getcwd(), 'embedding', self.conf.data_name, self.conf.pre_train))
            all_weights['user_embedding'] = tf.Variable(
                                pre_emb['user_embedding'], name='user_embedding')
            all_weights['item_embedding'] = tf.Variable(
                                pre_emb['item_embedding'], name='item_embedding')
            print('using pretrained initialization')


        if self.model_type == 'mlp':
            self.weight_size_list = [2 * self.emb_dim] + self.weight_size
        elif self.model_type == 'jrl':
            self.weight_size_list = [self.emb_dim] + self.weight_size
        else:
            self.weight_size_list = [3 * self.emb_dim] + self.weight_size

        for i in range(self.n_layers):
            all_weights['W_%d' %i] = tf.Variable(
                initializer([self.weight_size_list[i], self.weight_size_list[i+1]]), name='W_%d' %i)
            all_weights['b_%d' %i] = tf.Variable(
                initializer([1, self.weight_size_list[i+1]]), name='b_%d' %i)

        all_weights['h'] = tf.Variable(initializer([self.weight_size_list[-1], 1]), name='h')

        return all_weights

    def create_bpr_loss(self, users, pos_items, neg_items):
        pos_scores = self._create_inference(users, pos_items)
        self.prediction = pos_scores
        neg_scores = self._create_inference(users, neg_items)

        regularizer = tf.nn.l2_loss(tf.gather_nd(self.weights['user_embedding'], self.unique_user_list)) + tf.nn.l2_loss(tf.gather_nd(self.weights['item_embedding'], self.unique_item_list)) + tf.nn.l2_loss(tf.gather_nd(self.weights['item_embedding'], self.unique_item_neg_list))
        # regularizer = regularizer/self.batch_size

        maxi = tf.log(tf.nn.sigmoid(pos_scores - neg_scores))
        mf_loss = tf.negative(tf.reduce_sum(maxi, keepdims=True))

        emb_loss = self.regs * regularizer

        reg_loss = self.regs * tf.nn.l2_loss(self.weights['h'])

        return mf_loss, emb_loss, reg_loss

    def _create_inference(self, u_e, i_e):
        z = []

        if self.model_type == 'mlp':
            z.append(tf.concat([u_e, i_e], 1))
        elif self.model_type == 'jrl':
            z.append(u_e * i_e)
        else:
            z.append(tf.concat([u_e, i_e, u_e * i_e], 1))

        # z[0] = self.batch_norm_layer(z[0], train_phase=self.train_phase, scope_bn='bn_mlp')

        for i in range(self.n_layers):
            # (batch, W[i]) * (W[i], W[i+1]) + (1, W[i+1]) => (batch, W[i+1])
            # temp = self.batch_norm_layer(z[i], train_phase=self.train_phase, scope_bn='mlp_%d' % i)

            temp = tf.nn.relu(tf.matmul(z[i], self.weights['W_%d' % i]) + self.weights['b_%d' % i])
            temp = tf.nn.dropout(temp, self.dropout_keep[i])
            z.append(temp)

        agg_out = tf.matmul(z[-1], self.weights['h'])
        return agg_out

    def _create_all_ratings(self, u_e):
        z = []

        if self.model_type == 'jrl':
            u_1 = tf.expand_dims(u_e, axis=1)
            i_1 = tf.expand_dims(self.weights['item_embedding'], axis=0)
            z.append(tf.reshape(u_1 * i_1, [-1, self.emb_dim]))

        elif self.model_type == 'mlp':
            u_1 = tf.reshape(tf.tile(u_e, [1, self.n_items]), [-1, self.emb_dim])
            i_1 = tf.tile(self.weights['item_embedding'], [self.batch_size, 1])
            z.append(tf.concat([u_1, i_1], 1))
        else:
            u_1 = tf.expand_dims(u_e, axis=1)
            i_1 = tf.expand_dims(self.weights['item_embedding'], axis=0)
            u_i = tf.reshape(u_1 * i_1, [-1, self.emb_dim])

            u_1 = tf.reshape(tf.tile(u_e, [1, self.n_items]), [-1, self.emb_dim])
            i_1 = tf.tile(self.weights['item_embedding'], [self.batch_size, 1])
            z.append(tf.concat([u_1, i_1, u_i], 1))

        for i in range(self.n_layers):
            # (batch, W[i]) * (W[i], W[i+1]) + (1, W[i+1]) => (batch, W[i+1])
            z.append(tf.nn.relu(tf.matmul(z[i], self.weights['W_%d' % i]) + self.weights['b_%d' % i]))

        agg_out = tf.matmul(z[-1], self.weights['h']) # (batch, W[-1]) * (W[-1], 1) => (batch, 1)
        all_ratings = tf.reshape(agg_out, [-1, self.n_items])
        return all_ratings

    def _create_batch_ratings(self, u_e, i_e):
        z = []

        n_b_user = tf.shape(u_e)[0]
        n_b_item = tf.shape(i_e)[0]


        if self.model_type == 'jrl':
            u_1 = tf.expand_dims(u_e, axis=1)
            i_1 = tf.expand_dims(i_e, axis=0)
            z.append(tf.reshape(u_1 * i_1, [-1, self.emb_dim])) # (n_b_user * n_b_item, embed_size)

        elif self.model_type == 'mlp':
            u_1 = tf.reshape(tf.tile(u_e, [1, n_b_item]), [-1, self.emb_dim])
            i_1 = tf.tile(i_e, [n_b_user, 1])
            z.append(tf.concat([u_1, i_1], 1)) # (n_b_user * n_b_item, 2*embed_size)
        else:
            u_1 = tf.expand_dims(u_e, axis=1)
            i_1 = tf.expand_dims(i_e, axis=0)
            u_i = tf.reshape(u_1 * i_1, [-1, self.emb_dim])

            u_1 = tf.reshape(tf.tile(u_e, [1, n_b_item]), [-1, self.emb_dim])
            i_1 = tf.tile(i_e, [n_b_user, 1])
            z.append(tf.concat([u_1, i_1, u_i], 1))

        for i in range(self.n_layers):
            # (batch, W[i]) * (W[i], W[i+1]) + (1, W[i+1]) => (batch, W[i+1])
            z.append(tf.nn.relu(tf.matmul(z[i], self.weights['W_%d' % i]) + self.weights['b_%d' % i]))

        agg_out = tf.matmul(z[-1], self.weights['h']) # (batch, W[-1]) * (W[-1], 1) => (batch, 1)
        batch_ratings = tf.reshape(agg_out, [n_b_user, n_b_item])
        return batch_ratings

    def batch_norm_layer(self, x, scope_bn):
        with tf.variable_scope(scope_bn):
            return batch_norm(x, decay=0.9, center=True, scale=True, updates_collections=None,
            is_training=True, reuse=tf.AUTO_REUSE, trainable=True, scope=scope_bn)

    def saveVariables(self):
        self.saver = tf.train.Saver(max_to_keep=1)
    # def _statistics_params(self):
    #     # number of params
    #     total_parameters = 0
    #     for variable in self.weights.values():
    #         shape = variable.get_shape()  # shape is an array of tf.Dimension
    #         variable_parameters = 1
    #         for dim in shape:
    #             variable_parameters *= dim.value
    #         total_parameters += variable_parameters
    #     if self.verbose > 0:
    #         print("#params: %d" % total_parameters)