import tensorflow as tf
import numpy as np
import math


class Configuration:
    def __init__(self):
        self.vocab_size = None
        self.embedding_dim = 512

        self.multi_head_h = 8
        self.num_layer = 6

        self.k_dim = self.embedding_dim // self.multi_head_h
        self.v_dim = self.embedding_dim // self.multi_head_h

        self.max_input_length = 50
        self.max_output_length = 50

        self.batch_size = None


class Transformer:
    def __init__(self, vocab_size, config=Configuration()):

        self.config = config

        self.smoothing = True

        self.input_holder = tf.placeholder(tf.int32, shape=[None, None])
        self.input_mask_holder = tf.placeholder(tf.int32, shape=[None, None])
        self.output_holder = tf.placeholder(tf.int32, shape=[None, None])
        self.output_mask_holder = tf.placeholder(tf.int32, shape=[None, None])

        self.input = None
        self.output = None
        self.input_mask = None
        self.output_mask = None

        self.label = tf.placeholder(tf.float32, shape=[None, None, None])

        self.embedding_map = tf.get_variable('embedding_map', shape=[self.config.vocab_size, self.config.embedding_dim])

        self.dense = tf.layers.Dense(vocab_size, name="dense_out")

        self.out = None

        self.loss = None

        def build_positional_encoding(length):
            encoding = np.array([
                [pos / np.power(10000, 2 * (i // 2) / self.config.embedding_dim) for i in
                 range(self.config.embedding_dim)]
                if pos != 0 else np.zeros(self.config.embedding_dim) for pos in range(length)])

            encoding[1:, 0::2] = np.sin(encoding[1:, 0::2])
            encoding[1:, 1::2] = np.cos(encoding[1:, 1::2])
            return tf.convert_to_tensor(np.array([encoding for _ in range(self.config.batch_size)]), dtype=tf.float32)

        self.input_positional_encoding = build_positional_encoding(self.config.max_input_length)
        self.output_positional_encoding = build_positional_encoding(self.config.max_output_length)

    def build_embedding(self):

        def embedding_helper(x, x_encoding):
            return tf.nn.embedding_lookup(self.embedding_map * (self.config.embedding_dim ** 0.5), x) + x_encoding

        def masking_helper(lengths):
            return tf.sequence_mask(lengths, dtype=tf.float32)

        self.input = embedding_helper(self.input_holder, self.input_positional_encoding)
        self.output = embedding_helper(self.output_holder, self.output_positional_encoding)
        self.input_mask = masking_helper(self.input_mask_holder)
        self.output_mask = masking_helper(self.output_mask_holder)

    def build_encoder_layer(self):

        out = tf.multiply(self.input, self.input_mask)

        for num in range(self.config.num_layer):
            out = self.build_encoder_subLayer(out, "encoder_layer_" + str(num))

        return out

    def build_encoder_subLayer(self, inputs, layer_scope):

        multiHead_out = self.build_multiHead_attention(layer_scope, inputs, inputs, inputs)

        out = tf.contrib.layers.layer_norm(inputs + multiHead_out, axis=-1)

        FF_out = self.build_FFNN(layer_scope, out)

        return tf.contrib.layers.layer_norm(out + FF_out, axis=-1)

    def build_decoder_layer(self, encoder_output):

        out = tf.multiply(self.output, self.output_mask)

        for num in range(self.config.num_layer):
            out = self.build_decoder_sublayer(out, encoder_output, "decoder_layer_" + str(num))

        return out

    def build_decoder_sublayer(self, inputs, encoder_output, layer_scope):

        masked_multiHead_out = self.build_multiHead_attention(layer_scope + "_mask", inputs, inputs, inputs, mask=True)

        out = tf.contrib.layers.layer_norm(inputs + masked_multiHead_out, axis=-1)

        multiHead_out = self.build_multiHead_attention(layer_scope, out, encoder_output, encoder_output, mask=True)

        out = tf.contrib.layers.layer_norm(out + multiHead_out, axis=-1)

        FF_out = self.build_FFNN(layer_scope, out)

        return tf.contrib.layers.layer_norm(out + FF_out, axis=-1)

    def build_scaled_dotProduct_attention(self, Q, K, V, mask=False):

        Q_K = tf.matmul(Q, tf.transpose(K, perm=[0, 2, 1])) / (self.config.embedding_dim ** 0.5)

        if mask:
            diag_val = tf.ones_like(Q_K[0, :, :])

            tril = tf.linalg.LinearOperatorLowerTriangular(diag_val).to_dense()

            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(Q_K)[0], 1, 1])

            padding = tf.ones_like(masks) * (-math.pow(2, 32) + 1)

            Q_K = tf.where(tf.equal(masks, 0), padding, Q_K)

        out = tf.matmul(tf.nn.softmax(Q_K), V)

        return out

    def build_multiHead_attention(self, scope, Q, K, V, mask=False):

        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            heads = []

            for i in range(self.config.multi_head_h):
                dense_Q = tf.layers.Dense(self.config.k_dim, name="dense_Q_" + str(i))
                dense_K = tf.layers.Dense(self.config.k_dim, name="dense_K_" + str(i))
                dense_V = tf.layers.Dense(self.config.v_dim, name="dense_V_" + str(i))

                heads.append(self.build_scaled_dotProduct_attention(dense_Q(Q), dense_K(K), dense_V(V), mask=mask))

            multiHead = tf.concat(heads, axis=-1)

            dense_multiHead = tf.layers.Dense(self.config.embedding_dim, name="dense_multiHead")

        return dense_multiHead(multiHead)

    def build_FFNN(self, scope, inputs):

        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            dense_FF_1 = tf.layers.Dense(self.config.embedding_dim * 4, name="dense_FF1")
            dense_FF_2 = tf.layers.Dense(self.config.embedding_dim, name="dense_FF2")

        return dense_FF_2(tf.nn.relu(dense_FF_1(inputs)))

    def build(self):

        self.build_embedding()

        encoder_output = self.build_encoder_layer()

        out = self.build_decoder_layer(encoder_output)

        self.out = tf.nn.softmax(self.dense(out))

        def smoothing(inputs, epsilon=0.1):
            K = inputs.get_shape().as_list()[-1]
            return ((1 - epsilon) * inputs) + (epsilon / K)

        if self.smoothing:
            self.label = smoothing(self.label)

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.label, logits=self.out))
