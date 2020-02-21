import numpy as np
import random


class Configuration:
    def __init__(self):
        self.alpha = 0.0001
        self.num_epochs = 5
        self.M = 10 ** 8


class Node:
    def __init__(self, name=None, value=None):
        self.name = name
        self.val = value

        self.left = None
        self.right = None

        self.embedding_vector = random_weight()
        self.embedding_weight = random_weight()

        self.path = []


class Word2Vec:
    def __init__(self, tokens, vocab_dic, config=Configuration(), acc_mode=None, task_mode=None):

        self.config = config
        self.task_mode = task_mode
        self.acc_mode = acc_mode

        self.input_pairs = tokens

        self.embeddings = None

        if acc_mode == 'HierarchicalSoftmax':

            self._init_HuffmanTree(vocab_dic)

        else:
            self._init_SamplingWeight(vocab_dic)

    def _init_SamplingWeight(self, vocab_dic):

        self.embedding_weights = np.array([random_weight() for _ in range(len(vocab_dic))])
        self.embedding_vector = np.array([random_weight() for _ in range(len(vocab_dic))])

        count_sum = sum([count for word, count in vocab_dic])

        self.count_dic = {}
        for word, count in vocab_dic:
            self.count_dic[word] = count ** (3 / 4) / (count_sum ** (3 / 4))

        self.sampling_map = {}
        self.num_samples = 100
        self.sampling_keys = []

        self._build_sampling_map()

    def _build_sampling_map(self):

        index = 0
        for key, val in self.count_dic.items():
            self.sampling_map[str(index) + " " + str(index + int(self.config.M * val))] = key
            index += int(self.config.M * val)

        self.sampling_keys = [key for key in self.sampling_map.keys()]

    def _find_key(self, num):

        for key in self.sampling_keys:
            start = int(key.split()[0])
            end = int(key.split()[1])

            if start <= num < end:
                return self.sampling_map[key]

    def _build_sampling(self, index):
        indexes = []
        for i in range(self.num_samples):
            rand = random.randint(0, self.config.M - 1)
            k = self._find_key(rand)
            while k == index:
                rand = random.randint(0, self.config.M - 1)
                k = self._find_key(rand)
            indexes.append(k)
        return indexes

    def _init_HuffmanTree(self, vocab_dic):

        self.nodes_dic = {}

        self.nodes = []

        for word, count in vocab_dic:
            node = Node(word, count)
            self.nodes_dic[word] = node

            self.nodes.append(node)

        while len(self.nodes) > 1:
            self.nodes.sort(key=lambda n: n.val, reverse=True)

            node = Node(value=self.nodes[-1].val + self.nodes[-2].val)

            node.left = self.nodes.pop(-1)
            node.right = self.nodes.pop(-1)

            self.nodes.append(node)

        self.root = self.nodes[0]

        self._build_HuffmanTree_path(self.root)

        self.embedding_vector = {}

    def _build_HuffmanTree_path(self, node):

        if node.left:
            node.left.path = node.path[:] + [0]
            self._build_HuffmanTree_path(node.left)
        if node.right:
            node.right.path = node.path[:] + [1]
            self._build_HuffmanTree_path(node.right)

    def _get_node_path(self, word):
        return self.nodes_dic[word].path

    def _get_nodes(self, path, node, index):

        if index == len(path):
            return []

        nodes = []

        if path[index] == 0:
            nodes += [node.left] + self._get_nodes(path, node.left, index+1)
        else:
            nodes += [node.right] + self._get_nodes(path, node.right, index + 1)

        return nodes

    def _build_embedding_map(self, node):

        if node.left is None and node.right is None:
            self.embedding_vector[node.name] = node.emb
            return
        if node.left:
            self._build_embedding_map(node.left)
        if node.right:
            self._build_embedding_map(node.right)

    def train(self):

        for epoch in range(self.config.num_epochs):

            for pair in self.input_pairs:

                if self.acc_mode == 'HierarchicalSoftmax':
                    self._trian_HierarchicalSoftmax(self.task_mode, pair)
                else:
                    self._trian_NegativeSampling(self.task_mode, pair)

        if self.acc_mode == 'HierarchicalSoftmax':
            self._build_embedding_map(self.root)

            self.embeddings = np.array([self.embedding_vector[k] for k in sorted(self.embedding_vector)])
        else:
            self.embeddings = self.embedding_vector

    def _trian_HierarchicalSoftmax(self, mode, pair):

        x0 = pair[0]
        x_2c = pair[1:]

        x0_path = self._get_node_path(x0)
        x0_nodes = self._get_nodes(x0_path, self.root, 0)
        weights = np.array([node.embedding_weight for node in x0_nodes])

        if mode == 'Skip_Gram':
            embeddings = np.array([self.nodes_dic[word].embedding_vector for word in x_2c])

            f = sigmoid(np.dot(embeddings, weights.T))
            g = (1-np.array(x0_path).reshape((1, -1)) - f) * self.config.alpha
            e = np.dot(g, weights)
            weights += np.dot(g.T, embeddings)
            embeddings += e

            for num, node in enumerate(x0_nodes):
                node.embedding_weight = weights[num]
            for num, word in enumerate(x_2c):
                self.nodes_dic[word].embedding_vector = embeddings[num]
        else:
            x_sum = np.sum([self.nodes_dic[word].embedding_vector for word in x_2c], 0)

            f = sigmoid(np.dot(weights, x_sum))
            g = ((1 - np.array(x0_path) - f) * self.config.alpha).reshape((-1, 1))
            e = np.sum(g * weights, 0)
            weights += g * x_sum

            for num, node in enumerate(x0_nodes):
                node.weight = weights[num]
            for x in x_2c:
                self.nodes_dic[x].embedding_vector += e

    def _trian_NegativeSampling(self, mode, pair):

        x = pair[0]
        neg_samples = self._build_sampling(x)

        if mode == 'Skip_Gram':

            embeddings = self.embedding_vector[pair[1:]]

            f = sigmoid(np.dot(embeddings, self.embedding_weights[x]))
            g = ((1 - f) * self.config.alpha).reshape((-1, 1))
            e = g * self.embedding_weights[x]
            self.embedding_weights[x] += np.sum(g * embeddings, 0)

            f = sigmoid(np.dot(embeddings, self.embedding_weights[neg_samples].T))
            g = np.array(-f * self.config.alpha)
            e += np.dot(g, self.embedding_weights[neg_samples])
            self.embedding_weights[neg_samples] += np.dot(g.T, embeddings)
            self.embedding_vector[x] += np.sum(e, 0)

        else:
            embeddings = np.true_divide(np.sum(self.embedding_vector[pair[1:]], 0), len(pair) - 1)

            f = sigmoid(np.dot(self.embedding_weights[x], embeddings))
            g = (1 - f) * self.config.alpha
            e = g * self.embedding_weights[x]
            self.embedding_weights[x] += g * embeddings

            f = sigmoid(np.dot(self.embedding_weights[neg_samples], embeddings))
            g = np.array((-f * self.config.alpha)).reshape((-1, 1))
            e_neg = g * self.embedding_weights[neg_samples]
            e += np.sum(e_neg, 0)
            self.embedding_weights[neg_samples] += g * embeddings
            self.embedding_vector[neg_samples] += e


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def random_weight(dim=300):
    return np.random.normal(0, 0.1, dim).astype(np.float16)
