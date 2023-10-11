import gensim

class Word2VecLoader:
    def _init_(self, binary_file, limit=1000000):
        self.binary_file = binary_file
        self.limit = limit
        self.model = self.load_word2vec()

    def load_word2vec(self):
        return gensim.models.KeyedVectors.load_word2vec_format(self.binary_file, binary=True, limit=self.limit)

    def save_as_flat_file(self, output_file):
        self.model.save_word2vec_format(output_file, binary=False)
