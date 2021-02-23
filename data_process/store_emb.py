import tensorflow as tf
import os
import numpy as np
user_embedding = tf.Variable(tf.random_normal([24827, 100], stddev=0.01), name='user_embedding')
item_embedding = tf.Variable(tf.random_normal([16864, 100], stddev=0.01), name='item_embedding')
# user_social_embedding = tf.Variable(tf.random_normal([24827, 50], stddev=0.01), name='user_social_embedding')
# item_social_embedding = tf.Variable(tf.random_normal([16864, 50], stddev=0.01), name='item_social_embedding')
# W = [tf.layers.Dense(32, activation=tf.nn.leaky_relu, name='W0', use_bias=True), tf.layers.Dense(1, name='W1', use_bias=True)]
# x = W[1](W[0](tf.zeros([10, 96])))
variables_dict = {}
variables_dict['user_embedding'] = user_embedding
variables_dict['item_embedding'] = item_embedding
# variables_dict['user_social_embedding'] = user_social_embedding
# variables_dict['item_social_embedding'] = item_social_embedding
# for idx, W1 in enumerate(W):
#     variables_dict[f'W{idx}/kernel'] = W1.kernel
#     variables_dict[f'W{idx}/bias'] = W1.bias
saver = tf.train.Saver(variables_dict)
model_dir = 'BeiBei2_bpr_reg0.01_lr0.001_epoch0+1000_dim100_fs.ckpt'
checkpoint = 'pretrain/BeiBei2/{}'.format(model_dir)
tf_conf = tf.ConfigProto()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
tf_conf.gpu_options.allow_growth = True
sess = tf.Session(config=tf_conf)
saver.restore(sess, checkpoint)
user_embedding = user_embedding.eval(session=sess)
item_embedding = item_embedding.eval(session=sess)
# user_social_embedding = user_social_embedding.eval(session=sess)
# item_social_embedding = item_social_embedding.eval(session=sess)
# W = [(W1.kernel.eval(session=sess), W1.bias.eval(session=sess)) for W1 in W]

np.savez('embedding/BeiBei2/{}.npz'.format(model_dir), user_embedding=user_embedding, item_embedding=item_embedding)#, \
                                                        # user_social_embedding=user_social_embedding, item_social_embedding=item_social_embedding)
# np.savez('embedding/BeiBei2/{}_W.npz'.format(model_dir), W0_kernel=W[0][0], W0_bias=W[0][1], W1_kernel=W[1][0], W1_bias=W[1][1])
print('save done')