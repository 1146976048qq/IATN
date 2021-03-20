import tensorflow as tf
from utils import get_data_info, read_data, load_word_embeddings
from model import IATN
from evals import *
import os
import time
import math

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '4'

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
tf.enable_eager_execution(config=config)

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('emb_dim', 300, 'dimension of word embedding')
tf.app.flags.DEFINE_integer('n_hidden', 300, 'number of hidden unit')
tf.app.flags.DEFINE_integer('n_class', 3, 'number of distinct class')
tf.app.flags.DEFINE_float('l2_reg', 0.001, 'l2 regularization')

tf.app.flags.DEFINE_integer('max_a_len', 0, 'max length of aspects')
tf.app.flags.DEFINE_integer('max_s_len', 0, 'max length of sentences')
tf.app.flags.DEFINE_string('embedding_matrix', '', 'word ids to word vectors')

batch_size = 128
learning_rate = 0.01
n_epoch = 20
pre_processed = 1
embedding_file_name = 'data/kkzhang/glove.840B.300d.txt'
dataset = 'data/kkzhang/laptop/'
logdir = 'logs/'


def run(model, train_data, test_data):
	print("Train the IATN model ......")

	max_acc, step = 0., -1

    train_data_size = len(train_data[0])
    train_data = tf.data.Dataset.from_tensor_slices(train_data)
    train_data = train_data.shuffle(buffer_size=train_data_size).batch(batch_size, drop_remainder=True)

    test_data_size = len(test_data[0])
    test_data = tf.data.Dataset.from_tensor_slices(test_data)
    test_data = test_data.batch(batch_size, drop_remainder=True)

    iterator = tf.data.Iterator.from_structure(train_data.output_types, train_data.output_shapes)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    writer = tf.contrib.summary.create_file_writer(logdir)
    writer.set_as_default()

    for i in range(n_epoch):
    	cost, domain_predict_list, domain_label_list, sentiemnt_predict_list, sentiment_labels_list = 0., [], [], [], []
    	iterator.make_initializer(train_data)


        # for sentiment classification
        # for sentiment classification
        # for sentiment classification
    	for _ in range(math.floor(train_data_size / batch_size)):
    		data = iterator.get_next()
    		with tf.GradientTape() as tape:
    			domain_predict, sentiemnt_predict, domain_labels, sentiment_labels = model(data, dropout=0.2)

    			sentiemnt_loss_t = tf.nn.softmax_cross_entropy_with_logits_v2(
    									logits = sentiemnt_predict, sentiment_labels = sentiment_labels)
    			sentiment_loss = tf.reduce_mean(sentiemnt_loss_t)
                sentiment_cost += tf.reduce_sum(sentiemnt_loss_t)
            sentiment_grads = tape.gradient(sentiment_loss, model.variables)
            optimizer.apply_gradients(zip(sentiment_grads, model.variables))
            domain_predict_list.extend(tf.argmax(tf.nn.softmax(sentiemnt_predict), 1).numpy())
            domain_labels_list.extend(tf.argmax(sentiment_labels, 1).numpy())

        sentiment_train_acc, _, _ = evaluate(pred=sentiemnt_predict_list, gold=sentiment_labels_list)
        sentiment_train_loss = csentiment_cost / train_data_size
        tf.contrib.summary.scalar('sentiment_train_loss', sentiment_train_loss)
        tf.contrib.summary.scalar('sentiment_train_acc', sentiment_train_acc)


        # for domain classification
        # for domain classification
        # for domain classification
        for _ in range(math.floor(train_data_size / batch_size)):















