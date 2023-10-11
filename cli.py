import argparse
from word2vec_loader import Word2VecLoader
from similarity_calculator import SimilarityCalculator

def main():
    parser = argparse.ArgumentParser(description="Word2Vec and Phrase Similarity Calculator")

    parser.add_argument("--word2vec-file", required=True, help="C:\Users\I1925\Downloads\GoogleNews-vectors-negative300.bin")
    parser.add_argument("--phrases-file", required=True, help="phrases.csv")
    parser.add_argument("--output-similarity-matrix", required=True, help="vectors.csv")

    args = parser.parse_args()

    # Load Word2Vec vectors
    w2v_loader = Word2VecLoader(args.word2vec_file)

    # Calculate phrase similarities
    similarity_calculator = SimilarityCalculator(w2v_loader.model, args.phrases_file)
    similarity_matrix = similarity_calculator.calculate_similarity_matrix()

    # Save the similarity matrix as a CSV file
    pd.DataFrame(similarity_matrix, columns=similarity_calculator.phrases, index=similarity_calculator.phrases).to_csv(args.output_similarity_matrix)

if _name_ == "_main_":
    main()
