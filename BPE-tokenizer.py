import operator


class BPE:
    def __init__(self, tokens, vocab_size=9):

        self.vocab_size = vocab_size
        self.vocab = ['</w>']

        self._init_pairs(tokens)

        self.build_vocab()

    def _init_pairs(self, tokens):

        self.pairs = []
        self.dic = {}

        for token in tokens:
            for i in range(len(token) - 1):
                if token[i] not in self.vocab:
                    self.vocab.append(token[i])
                pair = token[i] + " " + token[i + 1]
                if pair in self.dic:
                    self.dic[pair] += 1
                else:
                    self.dic[pair] = 1
                self.pairs.append(pair)

            if token[-1] not in self.vocab:
                self.vocab.append(token[-1])

            pair = token[-1] + ' </w>'
            self.pairs.append(pair)
            if pair in self.dic:
                self.dic[pair] += 1
            else:
                self.dic[pair] = 1

    def _merge_pairs(self, token_pair):

        i = 0
        while i < len(self.pairs):
            if self.pairs[i] == token_pair:
                if i != 0 and self.pairs[i - 1].split()[-1] == token_pair.split()[0]:
                    start, end = self.pairs[i - 1].split()

                    if self.dic[self.pairs[i - 1]] == 1:
                        del self.dic[self.pairs[i - 1]]
                    else:
                        self.dic[self.pairs[i - 1]] -= 1

                    new_pair = start + " " + "".join(token_pair.split())
                    self.pairs[i - 1] = new_pair
                    if new_pair not in self.dic:
                        self.dic[new_pair] = 1
                    else:
                        self.dic[new_pair] += 1

                if i < len(self.pairs) - 1 and self.pairs[i + 1].split()[0] == token_pair.split()[-1]:
                    start, end = self.pairs[i + 1].split()

                    if self.dic[self.pairs[i + 1]] == 1:
                        del self.dic[self.pairs[i + 1]]
                    else:
                        self.dic[self.pairs[i + 1]] -= 1

                    new_pair = "".join(token_pair.split()) + " " + end
                    self.pairs[i + 1] = new_pair
                    if new_pair not in self.dic:
                        self.dic[new_pair] = 1
                    else:
                        self.dic[new_pair] += 1

                self.pairs.pop(i)
            else:
                i += 1

    def build_vocab(self):

        while len(self.vocab) != self.vocab_size and sum(self.dic.values()) != len(self.dic):
            sorted_dic = sorted(self.dic.items(), key=operator.itemgetter(1), reverse=True)

            token_pair = sorted_dic[0][0]
            self._merge_pairs(token_pair)

            del self.dic[token_pair]

            start, end = token_pair.split()
            startin = False
            endin = False

            for key in self.dic.keys():
                if start in key.split():
                    startin = True
                if end in key.split():
                    endin = True

            if not startin:
                self.vocab.remove(start)
            if not endin:
                self.vocab.remove(end)

            self.vocab.append(start + end)

        print(self.vocab)


bpe = BPE(['lower', 'lower',
           'low', 'low', 'low', 'low', 'low',
           'newest', 'newest', 'newest', 'newest', 'newest', 'newest',
           'widest', 'widest', 'widest'])
