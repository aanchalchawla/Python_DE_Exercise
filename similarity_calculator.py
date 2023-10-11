import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

class SimilarityCalculator:
    def _init_(self, word2vec_model, phrases_file):
        self.word2vec_model = word2vec_model
        self.phrases_df = pd.read_csv(phrases_file)
        self.phrases = self.phrases_df['Phrase'].tolist()

    def calculate_similarity_matrix(self):
        similarity_matrix = np.zeros((len(self.phrases), len(self.phrases)))

        for i, phrase1 in enumerate(self.phrases):
            for j, phrase2 in enumerate(self.phrases):
                # Calculate cosine similarity between phrase vectors
                similarity_matrix[i][j] = self.calculate_cosine_similarity(phrase1, phrase2)

        return similarity_matrix

    def calculate_cosine_similarity(self, phrase1, phrase2):
        tokens1 = phrase1.split()
        tokens2 = phrase2.split()

        # Calculate the average word vector for each phrase
        vector1 = np.mean([self.word2vec_model[word] for word in tokens1 if word in self.word2vec_model], axis=0)
        vector2 = np.mean([self.word2vec_model[word] for word in tokens2 if word in self.word2vec_model], axis=0)

        return cosine_similarity([vector1], [vector2])[0][0]
