import tensorflow as tf
import tensorflow.keras as keras


class Configuration:
    def __init__(self):

        self.vocab_size = None
        self.label_size = None

        self.embedding_dim = 128

        self.rnn_dim = 512

        self.num_heads = 8

        self.dim = self.rnn_dim // self.num_heads


class LAN(keras.Model):
    def __init__(self, config=Configuration()):
        super(LAN, self).__init__()

        self.config = config

        self.label_embedding = tf.Variable(tf.zeros([self.config.label_size, self.config.embedding_dim]))
        self.word_embedding = keras.layers.Embedding(self.vocab_size, self.config.embedding_dim, mask_zero=True)
        self.forward_LSTM = keras.layers.LSTM(self.config.rnn_dim, return_sequences=True)
        self.backward_LSTM = keras.layers.LSTM(self.config.rnn_dim, go_backwards=True, return_sequences=True)
        self.out_forward_LSTM = keras.layers.LSTM(self.config.rnn_dim, return_sequences=True)
        self.out_backward_LSTM = keras.layers.LSTM(self.config.rnn_dim, go_backwards=True, return_sequences=True)

        self.heads_weights = []
        for i in range(self.config.num_heads):
            dense_Q = keras.layers.Dense(
                self.config.dim, input_shape=(None, self.config.embedding_dim), name='dense_Q_' + str(i))
            dense_K = keras.layers.Dense(
                self.config.dim, input_shape=(None, self.config.embedding_dim), name='dense_K_' + str(i))
            dense_V = keras.layers.Dense(
                self.config.dim, input_shape=(None, self.config.embedding_dim), name='dense_V_' + str(i))
            self.heads_weights.append([dense_Q, dense_K, dense_V])

    def call(self, inputs):

        embed = self.word_embedding(inputs)

        embed_forward_out = self.forward_LSTM(embed)
        embed_backward_out = self.backward_LSTM(embed)

        H_out = embed_forward_out + embed_backward_out

        multihead_out = self.build_multiHead_attention(
            H_out, self.label_embedding, self.label_embedding, self.heads_weights)

        concat_H = tf.concat((H_out, multihead_out + H_out), axis=-1)

        forward_out = self.out_forward_LSTM(concat_H)
        backward_out = self.out_backward_LSTM(concat_H)

        H = forward_out + backward_out

        output = tf.matmul(H, tf.transpose(self.label_embedding, perm=[0, 2, 1])) / (self.config.embedding_dim ** 0.5)

        return keras.activations.softmax(output)

    def build_scaled_dotProduct_attention(self, Q, K, V):

        Q_K = tf.matmul(Q, tf.transpose(K, perm=[0, 2, 1])) / (self.config.embedding_dim ** 0.5)

        out = tf.matmul(keras.activations.softmax(Q_K), V)

        return out

    def build_multiHead_attention(self, Q, K, V, weights):

        heads = []

        for i in range(self.config.num_heads):
            dense_Q = weights[i][0]
            dense_K = weights[i][1]
            dense_V = weights[i][2]
            heads.append(self.build_scaled_dotProduct_attention(dense_Q(Q), dense_K(K), dense_V(V)))

        out = tf.concat(heads, axis=-1)

        return out
