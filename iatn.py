import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Bidirectional
# from attention import AttLayer

class IATN(tf.keras.Model):

    def __init__(self, config):
        super(IATN, self).__init__()

        self.embedding_dim = config.embedding_dim
        self.n_hidden = config.n_hidden
        self.n_class = config.n_class
        self.l2_reg = config.l2_reg

        self.max_aspect_len = config.max_aspect_len
        self.max_sentence_len = config.max_sentence_len
        self.embedding_matrix = config.embedding_matrix

        self.aspect_bilstm = tf.keras.layers.Bidirectional(LSTM(self.n_hidden,
                                                              return_sequences=True,
                                                              recurrent_initializer='glorot_uniform',
                                                              stateful=True))
        self.sentence_bilstm = tf.keras.layers.Bidirectional(LSTM(self.n_hidden,
                                                               return_sequences=True,
                                                               recurrent_activation='sigmoid',
                                                               recurrent_initializer='glorot_uniform',
                                                               stateful=True))

        self.aspect_w = tf.contrib.eager.Variable(tf.random_normal([self.n_hidden, self.n_hidden]), name='aspect_w')
        self.aspect_b = tf.contrib.eager.Variable(tf.zeros([self.n_hidden]), name='aspect_b')

        self.sentence_w = tf.contrib.eager.Variable(tf.random_normal([self.n_hidden, self.n_hidden]), name='sentence_w')
        self.sentence_b = tf.contrib.eager.Variable(tf.zeros([self.n_hidden]), name='sentence_b')

        self.output_fc = tf.keras.layers.Dense(self.n_class, kernel_regularizer=tf.keras.regularizers.l2(l=self.l2_reg))

    def call(self, data, dropout=0.5):
        aspects, sentences, labels, aspect_lens, sentence_lens = data
        aspect_inputs = tf.nn.embedding_lookup(self.embedding_matrix, aspects)
        aspect_inputs = tf.cast(aspect_inputs, tf.float32)
        aspect_inputs = tf.nn.dropout(aspect_inputs, keep_prob=dropout)

        sentence_inputs = tf.nn.embedding_lookup(self.embedding_matrix, sentences)
        sentence_inputs = tf.cast sentence_inputs, tf.float32)
        sentence_inputs = tf.nn.dropout sentence_inputs, keep_prob=dropout)

        aspect_outputs = self.aspect_bilstm(aspect_inputs)
        aspect_avg = tf.reduce_mean(aspect_outputs, 1)

        sentence_outputs = self.sentence_bilstm(sentence_inputs)
        sentence_avg = tf.reduce_mean sentence_outputs, 1)

        aspect_att = tf.nn.softmax(tf.nn.tanh(tf.einsum('ijk,kl,ilm->ijm', aspect_outputs, self.aspect_w,
                                                        tf.expand_dims sentence_avg, -1)) + self.aspect_b))
        aspect_rep = tf.reduce_sum(aspect_att * aspect_outputs, 1)

        sentence_att = tf.nn.softmax(tf.nn.tanh(tf.einsum('ijk,kl,ilm->ijm', sentence_outputs, self sentence_w,
                                                         tf.expand_dims(aspect_avg, -1)) + self sentence_b))
        sentence_rep = tf.reduce_sum sentence_att * sentence_outputs, 1)

        rep = tf.concat([aspect_rep, sentence_rep], 1)
        predict = self.output_fc(rep)

        return predict, labels
